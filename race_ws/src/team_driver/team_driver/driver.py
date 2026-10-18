#!/usr/bin/env python3
"""Refined driver: high speed on straights, fast sweepers, and smooth 180-deg hairpin navigation."""

import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from team_driver.pure_pursuit import PurePursuit, load_path
from team_driver.speed_controller import SpeedController

MAX_STEERING_RAD = 0.5236

DEFAULT_PARAMS = {
    'vehicle_ns': '/autodrive/roboracer_1',
    'max_range': 10.0,
    # Path
    'path_file': '/hackathon/maps/icra26_compete_centerline.csv',
    # Pure pursuit
    'lookahead_min': 0.72,              # [m] stable tracking on straights and wide curves
    'lookahead_hairpin': 0.44,          # [m] tight tracking strictly in the 180 hairpin apex
    'lookahead_gain': 0.26,             # [s] ld = min + gain * speed
    'lookahead_max': 2.2,               # [m]
    'wheelbase': 0.324,                 # [m]
    'max_steer_rad': 0.5236,            # [rad]
    # Speed Profile (Straights vs Sweeping Corners vs 180 Hairpin)
    'v_target': 5.5,                    # [m/s] top speed on straightaways
    'v_corner': 3.2,                    # [m/s] fast momentum through normal curves (was 2.6)
    'v_hairpin': 1.9,                  # [m/s] smooth, brisk pace through the 180 hairpin (was 1.7)
    'preview_dist': 2.8,                # [m] lookahead window for normal corners
    'hairpin_preview_dist': 2.8,        # [m] lookahead window for hairpin detection
    'corner_angle_thresh': 0.60,        # [rad] local bend threshold for normal curves
    'hairpin_angle_thresh': 1.60,       # [rad] (~92 deg) ONLY triggers on the true 180 U-turn!
    # Speed Controller (PI + Feed-Forward)
    'k_ff': 0.13,
    'kp': 0.14,
    'ki': 0.04,
    'integral_limit': 0.12,
    'throttle_min': 0.0,
    'throttle_max': 0.70,
    'overspeed_margin': 1.25,           # [m/s] smooth coasting in curves; prevents brake-jerking
    'brake_throttle': -0.22,            # decisive braking only when dropping from 5.0 to 1.8 m/s
    # Safety (LiDAR obstacle ahead)
    'safety_distance': 0.40,            # [m] emergency stop only; won't trigger on hairpin side ducts
    'safety_half_angle_deg': 3.5,       # [deg] tight forward cone
    'safety_speed': 1.4,
}

FALLBACK_MAP_DIR = '/opt/hackathon_maps'


def devkit_qos() -> QoSProfile:
    return QoSProfile(
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in DEFAULT_PARAMS}

        self.max_range = p['max_range']
        self.v_target = p['v_target']
        self.safety_distance = p['safety_distance']
        self.safety_half_angle = math.radians(p['safety_half_angle_deg'])
        self.safety_speed = p['safety_speed']
        self.v_corner = p['v_corner']
        self.preview_dist = p['preview_dist']
        self.corner_angle_thresh = p['corner_angle_thresh']
        self.v_hairpin = p['v_hairpin']
        self.hairpin_preview_dist = p['hairpin_preview_dist']
        self.hairpin_angle_thresh = p['hairpin_angle_thresh']
        self.lookahead_min_default = p['lookahead_min']
        self.lookahead_hairpin = p['lookahead_hairpin']

        path_file = self.resolve_path_file(p['path_file'])
        self.path = load_path(path_file)
        self.pursuit = PurePursuit(
            self.path,
            wheelbase=p['wheelbase'],
            max_steer_rad=p['max_steer_rad'],
            lookahead_min=p['lookahead_min'],
            lookahead_gain=p['lookahead_gain'],
            lookahead_max=p['lookahead_max'],
        )
        self.speed_ctl = SpeedController(
            k_ff=p['k_ff'], kp=p['kp'], ki=p['ki'],
            integral_limit=p['integral_limit'],
            throttle_min=p['throttle_min'], throttle_max=p['throttle_max'],
            overspeed_margin=p['overspeed_margin'],
            brake_throttle=p['brake_throttle'],
        )

        ns = str(p['vehicle_ns']).rstrip('/')
        qos = devkit_qos()

        self.throttle_pub = self.create_publisher(Float32, f'{ns}/throttle_command', qos)
        self.steering_pub = self.create_publisher(Float32, f'{ns}/steering_command', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(LaserScan, f'{ns}/lidar', self.scan_callback, qos)
        self.create_subscription(Odometry, f'{ns}/odom', self.odom_callback, qos)

        self.position = None
        self.yaw = 0.0
        self.speed = 0.0

        self.last_stamp = None
        self.dt = 0.025
        self.front_distance = self.max_range
        self.current_target = 0.0
        self._marker_divisor = 0
        self._log_divisor = 0
        self.get_logger().info(
            f'Pure pursuit on {path_file} ({len(self.path)} points), '
            f'v_target {self.v_target} m/s.')

    def resolve_path_file(self, path_file):
        if os.path.isfile(path_file):
            return path_file
        fallback = os.path.join(FALLBACK_MAP_DIR, os.path.basename(path_file))
        if os.path.isfile(fallback):
            self.get_logger().warn(f'{path_file} not found, using {fallback}')
            return fallback
        raise FileNotFoundError(f'path_file not found: {path_file}')

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def scan_callback(self, scan):
        self.dt = self.scan_dt(scan)
        ranges, angles = self.preprocess(scan)
        steering, throttle = self.plan(ranges, angles)
        self.publish(steering, throttle)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering * MAX_STEERING_RAD)

        self._log_divisor = (self._log_divisor + 1) % 40
        if self._log_divisor == 0:
            self.get_logger().info(
                f'v {self.speed:.2f}/{self.current_target:.2f} m/s  '
                f'throttle {throttle:+.3f}  steer {steering:+.2f}  '
                f'front {self.front_distance:.2f} m  idx {self.pursuit.nearest_idx}')

    def scan_dt(self, scan):
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        dt = stamp - self.last_stamp if self.last_stamp is not None else 0.0
        self.last_stamp = stamp
        return dt if 0.0 < dt < 0.2 else 0.025

    def preprocess(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    def preview_heading_change(self, start_idx, preview_dist):
        """Measure peak LOCAL curvature (over 5 waypoints ~0.6m) for normal corners."""
        remaining = preview_dist
        i = start_idx
        n = self.pursuit.n
        path = self.pursuit.path
        seg_len = self.pursuit.seg_len

        window = 5
        max_local_bend = 0.0

        while remaining > 0.0:
            d1 = path[(i + 2) % n] - path[i]
            d2 = path[(i + window + 2) % n] - path[(i + window) % n]
            yaw1 = math.atan2(d1[1], d1[0])
            yaw2 = math.atan2(d2[1], d2[0])
            local_bend = abs(math.atan2(math.sin(yaw2 - yaw1),
                                        math.cos(yaw2 - yaw1)))
            if local_bend > max_local_bend:
                max_local_bend = local_bend
            remaining -= seg_len[i]
            i = (i + 1) % n

        return max_local_bend

    def preview_total_bend(self, start_idx, preview_dist):
        """Measure true net reversal angle over a short arc to isolate the 180 hairpin."""
        remaining = preview_dist
        i = start_idx
        n = self.pursuit.n
        path = self.pursuit.path
        seg_len = self.pursuit.seg_len

        d0 = path[(i + 2) % n] - path[i]
        start_yaw = math.atan2(d0[1], d0[0])
        max_net_bend = 0.0

        while remaining > 0.0:
            d = path[(i + 2) % n] - path[i]
            seg_yaw = math.atan2(d[1], d[0])
            net_diff = abs(math.atan2(math.sin(seg_yaw - start_yaw),
                                      math.cos(seg_yaw - start_yaw)))
            if net_diff > max_net_bend:
                max_net_bend = net_diff
            remaining -= seg_len[i]
            i = (i + 1) % n

        return max_net_bend

    def plan(self, ranges, angles):
        """Pure pursuit with smooth 3-tier speed profile and zero discontinuities."""
        if self.position is None:
            self.current_target = 0.0
            return 0.0, 0.0

        x, y = self.position
        idx = self.pursuit.find_nearest(x, y)

        # 1. Preview curvature
        peak_local_bend = self.preview_heading_change(idx, self.preview_dist)
        hairpin_bend = self.preview_total_bend(idx, self.hairpin_preview_dist)

        # Continuous hairpin ratio (0.0 on normal curves, scaling to 1.0 ONLY in the 180 hairpin)
        hp_ratio = float(np.clip(
            (hairpin_bend - 1.25) / max(self.hairpin_angle_thresh - 1.25, 1e-3),
            0.0, 1.0
        ))

        # Smooth lookahead interpolation (eliminates sudden steering jumps)
        target_ld_min = self.lookahead_min_default - hp_ratio * (self.lookahead_min_default - self.lookahead_hairpin)
        self.pursuit.lookahead_min += 0.25 * (target_ld_min - self.pursuit.lookahead_min)

        steering = self.pursuit.steer(x, y, self.yaw, self.speed)

        # 2. Normal corner speed profile (stays fast between v_target=5.0 and v_corner=3.3)
        effective_bend = max(0.0, peak_local_bend - 0.18)
        bend_ratio = min(effective_bend / max(self.corner_angle_thresh - 0.18, 1e-3), 1.0)
        v_normal = self.v_target - (bend_ratio ** 1.3) * (self.v_target - self.v_corner)

        # 3. Apply hairpin target ONLY proportional to hp_ratio
        v_path = v_normal - hp_ratio * (v_normal - self.v_hairpin)

        # 4. Smooth steering-dependent target speed
        steer_mag = abs(steering)
        v_steer = self.v_target - (steer_mag ** 1.8) * (self.v_target - self.v_hairpin)
        raw_target = min(v_path, v_steer)

        # 5. Steer-aligned LiDAR check
        steer_angle_rad = steering * MAX_STEERING_RAD * 0.6
        front = np.abs(angles - steer_angle_rad) <= self.safety_half_angle
        valid = ranges[front][ranges[front] > 0.06]
        self.front_distance = float(np.percentile(valid, 15)) if valid.size else self.max_range

        if self.front_distance < self.safety_distance:
            raw_target = min(raw_target, self.safety_speed)

        # 6. Smooth target filter: brake decisively on entry (alpha=0.25), 
        # but accelerate smoothly on exit (alpha=0.08) to prevent rear wheel spin
        if self.current_target <= 0.0:
            self.current_target = raw_target
        else:
            alpha = 0.25 if raw_target < self.current_target else 0.08
            self.current_target += alpha * (raw_target - self.current_target)

        # 7. Traction Control: dynamically scale throttle_max based on steering angle
        # Full torque (1.0) when straight; capped to ~45% when wheels are turned hard
        traction_limit = 1.0 - 0.55 * (steer_mag ** 1.5)
        effective_throttle_max = self.speed_ctl.throttle_max * traction_limit

        # Temporarily apply traction limit to speed controller output
        orig_max = self.speed_ctl.throttle_max
        self.speed_ctl.throttle_max = effective_throttle_max
        throttle = self.speed_ctl.update(self.current_target, self.speed, self.dt)
        self.speed_ctl.throttle_max = orig_max

        return steering, throttle

    def publish(self, steering, throttle):
        self.steering_pub.publish(Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def publish_marker(self, target_angle):
        marker = Marker()
        marker.header.frame_id = 'roboracer_1'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)

        path = Marker()
        path.header.frame_id = 'world'
        path.header.stamp = marker.header.stamp
        path.ns = 'team_driver'
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.04
        path.color.r, path.color.g, path.color.a = 1.0, 0.6, 0.8
        path.points = [Point(x=float(px), y=float(py), z=0.0)
                       for px, py in np.vstack([self.path, self.path[:1]])]
        array.markers.append(path)

        if self.pursuit.lookahead_point is not None:
            target = Marker()
            target.header.frame_id = 'world'
            target.header.stamp = marker.header.stamp
            target.ns = 'team_driver'
            target.id = 2
            target.type = Marker.SPHERE
            target.action = Marker.ADD
            target.pose.position.x = float(self.pursuit.lookahead_point[0])
            target.pose.position.y = float(self.pursuit.lookahead_point[1])
            target.pose.orientation.w = 1.0
            target.scale.x = target.scale.y = target.scale.z = 0.2
            target.color.r, target.color.a = 1.0, 1.0
            array.markers.append(target)

        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()