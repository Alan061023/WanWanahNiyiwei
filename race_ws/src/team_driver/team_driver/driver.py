#!/usr/bin/env python3
"""ROBORACER AUTONOMOUS DRIVER - FOLLOW THE GAP (FTG) ALGORITHM

This node implements a Follow-the-Gap algorithm for reactive high-speed
obstacle avoidance and race track navigation.
"""

import math
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

# The steering command is normalised; this is what +-1.0 actually means.
MAX_STEERING_RAD = 0.5236


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

        # ROS 2 Parameters (Tunable via CLI or driver_params.yaml)
        self.declare_parameter('vehicle_ns', '/autodrive/roboracer_1')
        self.declare_parameter('max_range', 10.0)             # [m] clip scan max range
        self.declare_parameter('fov_deg', 140.0)              # [deg] total front field of view
        self.declare_parameter('bubble_radius', 0.45)         # [m] safety bubble around nearest obstacle
        self.declare_parameter('max_speed_throttle', 0.35)    # [-1, 1] speed on straightaways
        self.declare_parameter('min_speed_throttle', 0.15)    # [-1, 1] speed on sharp turns
        self.declare_parameter('kp', 0.85)                    # Steering proportional gain

        self.max_range = self.get_parameter('max_range').value
        self.fov_deg = self.get_parameter('fov_deg').value
        self.bubble_radius = self.get_parameter('bubble_radius').value
        self.max_speed_throttle = self.get_parameter('max_speed_throttle').value
        self.min_speed_throttle = self.get_parameter('min_speed_throttle').value
        self.kp = self.get_parameter('kp').value

        ns = str(self.get_parameter('vehicle_ns').value).rstrip('/')
        qos = devkit_qos()

        self.throttle_pub = self.create_publisher(Float32, f'{ns}/throttle_command', qos)
        self.steering_pub = self.create_publisher(Float32, f'{ns}/steering_command', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(LaserScan, f'{ns}/lidar', self.scan_callback, qos)
        self.create_subscription(Odometry, f'{ns}/odom', self.odom_callback, qos)
        self.create_subscription(Point, f'{ns}/ips', self.ips_callback, qos)

        # Telemetry state variables
        self.position = None      # (x, y) in the `world` frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self._marker_divisor = 0
        self.get_logger().info('Follow-the-Gap Driver Initialized and Ready!')

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def ips_callback(self, msg):
        self.position = (msg.x, msg.y)

    def scan_callback(self, scan):
    ranges, angles = self.preprocess(scan)
    steering, throttle = self.plan(ranges, angles)
    self.publish(steering, throttle)

    self._marker_divisor = (self._marker_divisor + 1) % 10
    if self._marker_divisor == 0:
        self.publish_marker(steering * MAX_STEERING_RAD)
        
        # Log status every 10 frames using valid local variables (steering & throttle)
        min_r = np.min(ranges) if len(ranges) > 0 else 0.0
        max_r = np.max(ranges) if len(ranges) > 0 else 0.0
        self.get_logger().info(
            f"Min range: {min_r:.2f}m | Max range: {max_r:.2f}m | "
            f"Steering: {steering:.2f} | Throttle: {throttle:.2f}"
        )

    def preprocess(self, scan):
    """Clean raw LiDAR scan ranges, crop to forward FOV, and construct beam angles."""
    # 1. Safely determine max range limit
    max_range = getattr(self, 'max_range', scan.range_max)
    min_range = scan.range_min if scan.range_min > 0 else 0.0

    # 2. Convert raw ranges to float64 array
    ranges = np.asarray(scan.ranges, dtype=np.float64)

    # 3. Replace inf and nan values with valid bounds
    ranges = np.nan_to_num(ranges, nan=0.0, posinf=max_range, neginf=0.0)
    ranges = np.clip(ranges, min_range, max_range)

    # 4. Compute angle array for all beams
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

    # 5. Filter array to forward 180° FOV (-90° to +90° / -1.57 to +1.57 rad)
    #    Prevents the car from trying to turn toward gaps behind or beside it.
    fov_mask = (angles >= -1.57) & (angles <= 1.57)
    ranges = ranges[fov_mask]
    angles = angles[fov_mask]

    return ranges, angles

    # ==================================================================
    # AUTONOMOUS DRIVING ALGORITHM (Follow the Gap)
    # ==================================================================
    def plan(self, ranges, angles):
        """Decide steering and throttle using Follow-the-Gap logic."""
        if len(ranges) == 0:
            return 0.0, 0.0

        # 1. Mask field of view (FOV) to focus on forward driving area
        half_fov = math.radians(self.fov_deg / 2.0)
        fov_mask = (angles >= -half_fov) & (angles <= half_fov)

        proc_ranges = np.copy(ranges)
        proc_ranges[~fov_mask] = 0.0

        if not np.any(proc_ranges > 0):
            return 0.0, self.min_speed_throttle

        # 2. Find the closest obstacle beam in the FOV
        valid_ranges = np.where(proc_ranges > 0, proc_ranges, np.inf)
        closest_idx = int(np.argmin(valid_ranges))
        closest_dist = valid_ranges[closest_idx]

        # 3. Apply safety bubble around the nearest obstacle point
        angle_increment = angles[1] - angles[0] if len(angles) > 1 else 0.004
        if closest_dist > 0.05:
            bubble_angle = math.atan2(self.bubble_radius, closest_dist)
            beam_radius = int(math.ceil(bubble_angle / abs(angle_increment)))

            start_idx = max(0, closest_idx - beam_radius)
            end_idx = min(len(proc_ranges), closest_idx + beam_radius + 1)
            proc_ranges[start_idx:end_idx] = 0.0

        # 4. Smooth scan data to filter single-beam noise
        window_size = 5
        kernel = np.ones(window_size) / window_size
        smoothed_ranges = np.convolve(proc_ranges, kernel, mode='same')

        # 5. Target the deepest gap point
        best_idx = int(np.argmax(smoothed_ranges))
        target_angle = angles[best_idx]

        # 6. Calculate normalized steering output [-1.0, 1.0]
        steering = self.kp * (target_angle / MAX_STEERING_RAD)
        steering = float(np.clip(steering, -1.0, 1.0))

        # 7. Calculate dynamic speed (slow down on sharp turns, accelerate on straights)
        steering_severity = abs(steering)
        front_beam_idx = len(proc_ranges) // 2
        front_distance = ranges[front_beam_idx]

        if front_distance < 0.5:
            # Emergency crawl if barrier is close directly ahead
            throttle = 0.05
        else:
            throttle = self.max_speed_throttle - (
                steering_severity * (self.max_speed_throttle - self.min_speed_throttle)
            )

        return steering, float(throttle)

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