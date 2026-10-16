#!/usr/bin/env python3
"""Baseline driver: pure pursuit on the centreline, constant target speed.

This is the node the judges run: `ros2 run team_driver driver`.

    scan_callback (40 Hz)  ->  preprocess  ->  plan  ->  publish

plan() steers with pure pursuit (pure_pursuit.py) along the path loaded from
`path_file` - by default the organisers' traced centreline,
maps/icra26_compete_centerline.csv - using ground-truth pose from /odom. A
speed controller (speed_controller.py) turns a constant `v_target` into a
throttle, and the target is capped to `safety_speed` when the LiDAR sees
something closer than `safety_distance` straight ahead.

Every tunable value is a ROS parameter, set in config/driver_params.yaml and
defaulted in DEFAULT_PARAMS below; keep the two in step.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

--------------------------------------------------------------------------
THE CONTROL INTERFACE IS NOT A SPEED REQUEST
--------------------------------------------------------------------------
AutoDRIVE takes a normalised throttle, not a target speed:

    /autodrive/roboracer_1/throttle_command   Float32, [-1, 1]
    /autodrive/roboracer_1/steering_command   Float32, [-1, 1]  ->  +-0.5236 rad

-1 is full reverse, 0 is coasting, +1 is full torque. There is no speed
controller between you and the motor, so "go at 3 m/s" is a control problem you
now own. Whatever algorithm you pick will decide a *target* speed; turning that
into a throttle is a separate job, and it is worth several seconds a lap. See
`docs/04-algorithms.md` §4.1.

--------------------------------------------------------------------------
WHAT YOU MAY READ  (see docs/06-rules.md)
--------------------------------------------------------------------------
    .../lidar            1080 beams over 270 deg, 0.06-10 m
    .../imu              orientation, angular velocity, linear acceleration
    .../left_encoder     wheel angle, 16 PPR x 120
    .../right_encoder
    .../front_camera     RGB image (headless runs do not produce one)
    .../throttle         actuator feedback
    .../steering
    .../ips              ground-truth position - ALLOWED here, see below
    .../odom             ground-truth pose and velocity - ALLOWED, RECOMMENDED
    .../lap_count        .../lap_time  .../last_lap_time  .../best_lap_time
    .../collision_count
    /tf, /tf_static

The AutoDRIVE competition rules mark the pose and race-telemetry topics
"restricted": legal for development, not at race time. **The hackathon does not
adopt that restriction for reading.** Every input above is yours to use during
a scored run, ground-truth pose included. What stays restricted is *publishing*
`/autodrive/reset_command`, which is the referee's (rule 31).
"""

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

# The steering command is normalised; this is what +-1.0 actually means.
MAX_STEERING_RAD = 0.5236

# Every tunable value, with its default. These defaults MUST match
# config/driver_params.yaml: a judged run may start the node with no parameter
# file at all (evaluate.launch.py only passes one if asked), and then these are
# what drives the car.
DEFAULT_PARAMS = {
    'vehicle_ns': '/autodrive/roboracer_1',
    'max_range': 10.0,                  # [m] clip the scan here
    # Path
    'path_file': '/hackathon/maps/icra26_compete_centerline.csv',
    # Pure pursuit
    'lookahead_min': 0.6,               # [m]
    'lookahead_gain': 0.15,             # [s]  ld = min + gain * speed
    'lookahead_max': 1.5,               # [m]
    'wheelbase': 0.324,                 # [m]
    'max_steer_rad': 0.5236,            # [rad] what steering_command = 1.0 means
    # Speed
    'v_target': 1.5,                    # [m/s] constant for now
    'k_ff': 0.05,                       # [throttle per m/s] feed-forward
    'kp': 0.10,                         # [throttle per m/s]
    'ki': 0.05,                         # [throttle per m]
    'integral_limit': 0.10,             # [throttle] max contribution of the I term
    'throttle_min': 0.0,                # lift off, never brake, when slightly fast
    'throttle_max': 0.30,
    'overspeed_margin': 1.0,            # [m/s] this much too fast allows braking
    'brake_throttle': -0.10,            # floor when badly too fast
    # Safety
    'safety_distance': 0.8,             # [m] LiDAR, +-safety_half_angle_deg ahead
    'safety_half_angle_deg': 10.0,
    'safety_speed': 0.5,                # [m/s] target speed cap when too close
}

# Docker copies maps/ here too, in case /hackathon is not where the repo is.
FALLBACK_MAP_DIR = '/opt/hackathon_maps'


def devkit_qos() -> QoSProfile:
    """Match the QoS the AutoDRIVE bridge publishes and subscribes with.

    RELIABLE / KEEP_LAST(1) / VOLATILE. Get this wrong and the topics simply do
    not connect: a BEST_EFFORT subscriber will never see a RELIABLE publisher's
    messages, and `ros2 topic list` will still cheerfully show the topic.
    """
    return QoSProfile(
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p v_target:=1.0
        # config/driver_params.yaml sets them all.
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in DEFAULT_PARAMS}

        self.max_range = p['max_range']
        self.v_target = p['v_target']
        self.safety_distance = p['safety_distance']
        self.safety_half_angle = math.radians(p['safety_half_angle_deg'])
        self.safety_speed = p['safety_speed']

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

        # Latest known pose and speed. Ground truth from the simulator, which
        # the hackathon rules allow you to use - so use it. Only /odom is
        # used: /ips reports the IPS sensor, 0.08 m ahead of the rear axle, and
        # mixing the two would make the position jump between two points.
        self.position = None      # (x, y) in the `world` frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self.last_stamp = None    # [s] of the previous scan, for the PI's dt
        self.dt = 0.025           # [s]
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

    # ------------------------------------------------------------------
    # Where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (40 Hz).
    # ------------------------------------------------------------------
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
        """Seconds since the previous scan, by the simulator's clock.

        The bridge runs lock-step with the simulator, which need not keep up
        with wall time, so the scan stamps are the honest clock. Anything
        implausible falls back to the nominal 40 Hz.
        """
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        dt = stamp - self.last_stamp if self.last_stamp is not None else 0.0
        self.last_stamp = stamp
        return dt if 0.0 < dt < 0.2 else 0.025

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """Pure pursuit on the centreline at a constant target speed.

        Returns `(steering, throttle)`, both normalised to [-1, 1]. Steering
        is a fraction of full lock (MAX_STEERING_RAD); throttle is torque, not
        speed.
        """
        self.current_target = 0.0
        if self.position is None:
            return 0.0, 0.0     # no pose yet: stay put

        x, y = self.position
        steering = self.pursuit.steer(x, y, self.yaw, self.speed)

        # Safety: cap the speed if something is close straight ahead.
        front = np.abs(angles) <= self.safety_half_angle
        valid = ranges[front][ranges[front] > 0.06]     # 0 means "no reading"
        self.front_distance = float(valid.min()) if valid.size else self.max_range

        v_target = self.v_target
        if self.front_distance < self.safety_distance:
            v_target = min(v_target, self.safety_speed)
        self.current_target = v_target

        throttle = self.speed_ctl.update(v_target, self.speed, self.dt)
        return steering, throttle

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, throttle):
        self.steering_pub.publish(Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
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

        # The path being followed, as a closed loop in the world frame.
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

        # Where pure pursuit is aiming.
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
