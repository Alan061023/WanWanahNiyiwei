#!/usr/bin/env python3
"""High-speed driver: curvature-profiled pure pursuit with predictive braking.

Drop-in replacement for the baseline `ros2 run team_driver driver`.
"""

import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from team_driver.pure_pursuit import PurePursuit, load_path
from team_driver.speed_controller import SpeedController

MAX_STEERING_RAD = 0.5236

# Keep in sync with config/driver_params.yaml
DEFAULT_PARAMS = {
    'vehicle_ns': '/autodrive/roboracer_1',
    'max_range': 10.0,                  # [m] clip the scan here
    # Path
    'path_file': '/hackathon/maps/icra26_compete_centerline.csv',
    # Pure pursuit (tuned for higher speeds)
    'lookahead_min': 0.75,              # [m] tighter low-speed cornering
    'lookahead_gain': 0.22,             # [s] ld = min + gain * speed
    'lookahead_max': 2.40,              # [m] stability on fast straights
    'wheelbase': 0.324,                 # [m]
    'max_steer_rad': 0.5236,            # [rad]
    # Speed & Curvature Profile
    'v_target': 5.0,                    # [m/s] top speed on straights
    'v_min': 1.6,                       # [m/s] minimum apex speed
    'a_lat_max': 4.2,                   # [m/s^2] max lateral cornering accel
    'a_brake_max': 4.5,                 # [m/s^2] backward-pass braking decel
    'a_accel_max': 3.5,                 # [m/s^2] forward-pass acceleration
    'brake_lookahead_m': 0.65,          # [m] look ahead on speed profile for motor lag
    'steer_slowdown': 0.25,             # speed reduction factor at full steering lock
    # Speed Controller (uncapped torque & firmer braking)
    'k_ff': 0.12,                       # [throttle per m/s] feed-forward
    'kp': 0.35,                         # [throttle per m/s]
    'ki': 0.05,                         # [throttle per m]
    'integral_limit': 0.15,             # [throttle] max contribution of the I term
    'throttle_min': -0.15,              # light drag braking on minor overspeed
    'throttle_max': 0.85,               # unlocked from 0.30 for fast straights
    'overspeed_margin': 0.25,           # [m/s] trigger hard braking quickly before turns
    'brake_throttle': -0.65,            # [throttle] firm braking into hairpins
    # Safety
    'safety_distance': 0.55,            # [m] LiDAR cone along steering direction
    'safety_half_angle_deg': 8.0,
    'safety_speed': 0.8,                # [m/s] target speed cap when blocked
}

FALLBACK_MAP_DIR = '/opt/hackathon_maps'


def devkit_qos() -> QoSProfile:
    """Match the QoS the AutoDRIVE bridge publishes and subscribes with."""
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

        self.max_range = float(p['max_range'])
        self.v_target = float(p['v_target'])
        self.v_min = float(p['v_min'])
        self.a_lat_max = float(p['a_lat_max'])
        self.a_brake_max = float(p['a_brake_max'])
        self.a_accel_max = float(p['a_accel_max'])
        self.brake_lookahead_m = float(p['brake_lookahead_m'])
        self.steer_slowdown = float(p['steer_slowdown'])
        self.max_steer_rad = float(p['max_steer_rad'])

        self.safety_distance = float(p['safety_distance'])
        self.safety_half_angle = math.radians(float(p['safety_half_angle_deg']))
        self.safety_speed = float(p['safety_speed'])

        path_file = self.resolve_path_file(p['path_file'])
        self.path = load_path(path_file)

        # Precompute curvature-based velocity profile along the closed loop
        self.speed_profile, self.mean_ds = self.compute_speed_profile(self.path)
        self.brake_lookahead_pts = max(
            1, int(round(self.brake_lookahead_m / max(self.mean_ds, 1e-3)))
        )

        self.pursuit = PurePursuit(
            self.path,
            wheelbase=p['wheelbase'],
            max_steer_rad=p['max_steer_rad'],
            lookahead_min=p['lookahead_min'],
            lookahead_gain=p['lookahead_gain'],
            lookahead_max=p['lookahead_max'],
        )
        self.speed_ctl = SpeedController(
            k_ff=p['k_ff'],
            kp=p['kp'],
            ki=p['ki'],
            integral_limit=p['integral_limit'],
            throttle_min=p['throttle_min'],
            throttle_max=p['throttle_max'],
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
        self._cached_angles = None
        self._marker_divisor = 0
        self._log_divisor = 0

        # Pre-build static path points once to avoid reallocating in the 40 Hz loop
        closed_path = np.vstack([self.path[:, :2], self.path[:1, :2]])
        self._static_path_points = [
            Point(x=float(px), y=float(py), z=0.0) for px, py in closed_path
        ]

        self.get_logger().info(
            f'Fast pure pursuit on {path_file} ({len(self.path)} pts), '
            f'speed profile [{self.speed_profile.min():.2f}, '
            f'{self.speed_profile.max():.2f}] m/s.'
        )

    def resolve_path_file(self, path_file):
        if os.path.isfile(path_file):
            return path_file
        fallback = os.path.join(FALLBACK_MAP_DIR, os.path.basename(path_file))
        if os.path.isfile(fallback):
            self.get_logger().warn(f'{path_file} not found, using {fallback}')
            return fallback
        raise FileNotFoundError(f'path_file not found: {path_file}')

    def compute_speed_profile(self, path):
        """Compute a smooth cornering + braking velocity profile over the loop."""
        pts = np.asarray(path[:, :2], dtype=np.float64)
        n = len(pts)

        # Segment lengths around the closed loop
        diffs = np.roll(pts, -1, axis=0) - pts
        ds = np.hypot(diffs[:, 0], diffs[:, 1])
        mean_ds = float(np.mean(ds)) if n > 0 else 0.1

        # Use a ~0.35 m chord stride so discretization noise doesn't spike curvature
        stride = max(1, int(round(0.35 / max(mean_ds, 1e-3))))
        p_prev = np.roll(pts, stride, axis=0)
        p_next = np.roll(pts, -stride, axis=0)

        a = pts - p_prev
        b = p_next - pts
        c = p_next - p_prev

        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        denom = (
            np.hypot(a[:, 0], a[:, 1])
            * np.hypot(b[:, 0], b[:, 1])
            * np.hypot(c[:, 0], c[:, 1])
            + 1e-9
        )
        curvature = 2.0 * cross / denom

        # Smooth curvature with a small moving window to handle chicane transitions
        kernel_size = max(3, int(round(0.5 / max(mean_ds, 1e-3))))
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        padded_k = np.pad(curvature, pad, mode='wrap')
        smooth_k = np.convolve(
            padded_k, np.ones(kernel_size) / kernel_size, mode='valid'
        )

        # 1. Lateral acceleration limit: v = sqrt(a_lat_max / curvature)
        v_prof = np.sqrt(self.a_lat_max / np.maximum(smooth_k, 1e-5))
        v_prof = np.clip(v_prof, self.v_min, self.v_target)

        # 2. Backward pass (2 laps for wrap-around): brake before entering corners
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                nxt = (i + 1) % n
                v_allow = math.sqrt(
                    v_prof[nxt] * v_prof[nxt] + 2.0 * self.a_brake_max * ds[i]
                )
                if v_allow < v_prof[i]:
                    v_prof[i] = v_allow

        # 3. Forward pass (2 laps for wrap-around): smooth throttle ramp-up
        for _ in range(2):
            for i in range(n):
                nxt = (i + 1) % n
                v_allow = math.sqrt(
                    v_prof[i] * v_prof[i] + 2.0 * self.a_accel_max * ds[i]
                )
                if v_allow < v_prof[nxt]:
                    v_prof[nxt] = v_allow

        return v_prof, mean_ds

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
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
                f'front {self.front_distance:.2f} m  idx {self.pursuit.nearest_idx}'
            )

    def scan_dt(self, scan):
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        dt = stamp - self.last_stamp if self.last_stamp is not None else 0.0
        self.last_stamp = stamp
        return dt if 0.0 < dt < 0.2 else 0.025

    def preprocess(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        np.nan_to_num(ranges, copy=False, nan=0.0, posinf=self.max_range, neginf=0.0)
        np.clip(ranges, 0.0, self.max_range, out=ranges)

        if self._cached_angles is None or len(self._cached_angles) != len(ranges):
            self._cached_angles = (
                scan.angle_min + np.arange(len(ranges), dtype=np.float64) * scan.angle_increment
            )
        return ranges, self._cached_angles

    def plan(self, ranges, angles):
        """Curvature-profiled pure pursuit with steer-aligned obstacle check."""
        self.current_target = 0.0
        if self.position is None:
            return 0.0, 0.0

        x, y = self.position
        steering = self.pursuit.steer(x, y, self.yaw, self.speed)

        # Look up the minimum profile speed over the immediate braking horizon
        idx = self.pursuit.nearest_idx
        n = len(self.speed_profile)
        horizon_indices = (idx + np.arange(self.brake_lookahead_pts + 1)) % n
        v_profile = float(np.min(self.speed_profile[horizon_indices]))

        # Modulate slightly when steering hard to prevent rear-end breakaway
        steer_factor = max(0.60, 1.0 - self.steer_slowdown * abs(steering))
        v_target = max(self.v_min, v_profile * steer_factor)

        # Aim the safety LiDAR cone into the turn so outer corner walls don't
        # false-trigger the safety speed cap on corner entry
        cone_center = steering * self.max_steer_rad * 0.65
        front = np.abs(angles - cone_center) <= self.safety_half_angle
        front_ranges = ranges[front]
        valid = front_ranges[front_ranges > 0.06]
        self.front_distance = float(valid.min()) if valid.size else self.max_range

        # Scale safety distance slightly with current speed
        dynamic_safety_dist = self.safety_distance + 0.10 * self.speed
        if self.front_distance < dynamic_safety_dist:
            v_target = min(v_target, self.safety_speed)

        self.current_target = v_target
        throttle = self.speed_ctl.update(v_target, self.speed, self.dt)
        return steering, throttle

    def publish(self, steering, throttle):
        self.steering_pub.publish(Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def publish_marker(self, target_angle):
        if self.marker_pub.get_subscription_count() == 0:
            return

        now_msg = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = 'roboracer_1'
        marker.header.stamp = now_msg
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
        path.header.stamp = now_msg
        path.ns = 'team_driver'
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.04
        path.color.r, path.color.g, path.color.a = 1.0, 0.6, 0.8
        path.points = self._static_path_points
        array.markers.append(path)

        if self.pursuit.lookahead_point is not None:
            target = Marker()
            target.header.frame_id = 'world'
            target.header.stamp = now_msg
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