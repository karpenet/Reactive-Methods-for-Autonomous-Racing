#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import sys
sys.path.append("/sim_ws/src/gap_follow/scripts/")

try:
    from cubic_spline_planner import calc_spline_course
except ImportError:
    raise

#  Constants from xacro
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('reactive_node')

        # Params
        self.declare_parameter('window_size', 30)
        self.declare_parameter('max_horizon', 5.0) # unused

        self.declare_parameter('angle_resolution', np.deg2rad(5.0))
        self.declare_parameter('radius_resolution', 0.2)
        self.declare_parameter('min_radius', 0.2)
        self.declare_parameter('max_radius', 2.0)

        self.declare_parameter('car_width', (WIDTH + 2 * WHEEL_LENGTH))
        self.declare_parameter('collision_check_resolution', 0.1)
        self.declare_parameter('dist_weight', 1.0)
        self.declare_parameter('angle_weight', 6.5)

        self.declare_parameter('kp', 0.30)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 1.6)
        self.declare_parameter("max_control", MAX_STEER)

        self.declare_parameter("err_low_tol", np.deg2rad(0.0))
        self.declare_parameter("err_high_tol", np.deg2rad(90.0))
        self.declare_parameter("low_vel", 0.5)
        self.declare_parameter("mid_vel", 2.0)
        self.declare_parameter("high_vel", 4.0)
        self.declare_parameter("velocity_attenuation",3.5)
        
        self.prev_steer = 0.0
        self.declare_parameter("steer_alpha",1.0)

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0

        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub_ = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def preprocess_lidar(self, ranges, range_min=0.0, range_max=np.inf):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (e.g. > 3m)
        """
        # Remove invalid readings
        proc_ranges = np.clip(ranges, range_min, range_max)

        # Clip high values
        # max_horizon = self.get_parameter('max_horizon').get_parameter_value().double_value
        # proc_ranges = np.clip(proc_ranges, 0, max_horizon)

        # Average over window
        window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        window = np.ones(window_size) / window_size
        proc_ranges = ndimage.convolve1d(proc_ranges, window, mode='nearest')

        return proc_ranges

    def sample_nodes(self, ranges, angles):
        """ Sample candidate points in front of car
        """

        # 1. Sample points in polar coordinate system
        radius_resolution = self.get_parameter('radius_resolution').get_parameter_value().double_value
        max_radius = self.get_parameter('max_radius').get_parameter_value().double_value
        min_radius = self.get_parameter('min_radius').get_parameter_value().double_value
        num_radius = int((max_radius - min_radius) / radius_resolution) + 1
        r = np.linspace(min_radius, max_radius, num_radius)

        angle_resolution = self.get_parameter('angle_resolution').get_parameter_value().double_value
        max_angle = np.pi / 2
        min_angle = -np.pi / 2
        num_angles = int((max_angle - min_angle) / angle_resolution) + 1
        a = np.linspace(min_angle, max_angle, num_angles)

        rr, aa = np.meshgrid(r, a)

        # 2. Filter candidate points such that:
        #   (I)  hit obstacle: rho[theta] > ranges[theta]
        indices = ((a - angles[0]) / (angles[1] - angles[0])).astype(int)
        max_dist = ranges[indices]
        valid_mask = (max_dist.reshape(-1, 1) - rr > 0)
        rho_p, theta_p = rr[valid_mask], aa[valid_mask]

        #   (II) too close to obstacle: dist(rho[theta], ranges[some index]) > safe dist
        car_width = self.get_parameter('car_width').get_parameter_value().double_value
        x_p, y_p = rho_p * np.cos(theta_p), rho_p * np.sin(theta_p)  # x_p (n,)  y_p (n,)
        x_s, y_s = ranges * np.cos(angles), ranges * np.sin(angles)  # x_s (m,)  y_s (m,)
        x_diff = x_s - x_p.reshape(-1, 1)  # x_diff (n, m)
        y_diff = y_s - y_p.reshape(-1, 1)  # y_diff (n, m)
        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)  # dist (n, m)
        valid_indices = np.all(dist > car_width, axis=1)  # valid_indices (n,)
        rho_p, theta_p = rho_p[valid_indices], theta_p[valid_indices]  # rho_p (nv,)  theta_p (nv,)
        x_p, y_p = x_p[valid_indices], y_p[valid_indices]  # x_p (nv,)  y_p (nv,)
        dist = dist[valid_indices]  # dist (nv, m)

        # Sort output based on rho
        sort_indices = np.argsort(rho_p)
        rho_p = rho_p[sort_indices]
        theta_p = theta_p[sort_indices]
        x_p = x_p[sort_indices]
        y_p = y_p[sort_indices]
        dist = dist[sort_indices]

        return rho_p, theta_p, x_p, y_p, dist

    @staticmethod
    def calc_potential(dist):
        """ Calculate potential for each candidate point
        """
        potentials = -np.sum(1 / dist ** 2, axis=1)  # potentials (nv,)
        return potentials

    @staticmethod
    def get_path(rho, score):
        """ Select best score indices for each rho
        """
        path_indices = []
        prev_rho = -np.inf
        for i in range(len(rho)):
            curr_rho, curr_score = rho[i], score[i]
            if curr_rho > prev_rho:
                path_indices.append(i)
            elif curr_score > score[path_indices[-1]]:
                path_indices[-1] = i
            prev_rho = curr_rho
        path_indices = np.array(path_indices)

        return path_indices

    def path_smoother(self, path_x, path_y, ranges, angles):
        """ Remove redundant waypoints to smooth path
        """
        path = np.vstack((path_x, path_y)).T  # path (n, 2)
        n = len(path)

        # Construct dynamic programming table with size (n,)
        # dp[i] represents minimum cost from node 0 to node i

        # Initialization:
        #     dp[i] = 0   for i == 0
        #     dp[i] = +∞  for i != 0
        dp = np.full((n,), np.inf)
        dp[0] = 0

        # Also keep track of each node's parent
        # Initialize parent[i + 1] = i
        parents = {}
        for i in range(n - 1):
            parents[i + 1] = i

        # Iterate node index to fill dp table
        for i in range(1, n):
            # If two nodes can be connected directly
            if not self.has_collision(path[0], path[i], ranges, angles):
                dp[i] = np.linalg.norm(path[0] - path[i])  # cost equal to nodes distance
                parents[i] = 0  # update node i parent
                continue

            # If two nodes cannot be connected directly,
            # then dp[i] = min{dp[j] + d(j, i)} for 0 < j < i
            for j in range(1, i):
                if self.has_collision(path[j], path[i], ranges, angles):
                    continue  # cannot connect j and i directly
                cost = dp[j] + self.calc_cost(path, j, i, parents)
                if cost >= dp[i]:
                    continue
                dp[i] = cost
                parents[i] = j

        # Back track final path
        smooth_path = []
        node_idx = n - 1
        while node_idx != 0:
            smooth_path.append(path[node_idx])
            node_idx = parents[node_idx]
        smooth_path.append(path[0])
        smooth_path = np.flipud(np.array(smooth_path))

        return smooth_path[:, 0], smooth_path[:, 1]

    def has_collision(self, point_1, point_2, ranges, angles):
        resolution = self.get_parameter('collision_check_resolution').get_parameter_value().double_value
        dist = np.linalg.norm(point_1 - point_2)
        num_points = int(dist / resolution) + 1
        points = np.linspace(point_1, point_2, num_points)
        points = points[1:-1]  # start and end points have been checked safe

        car_width = self.get_parameter('car_width').get_parameter_value().double_value
        x_p, y_p = points[:, 0], points[:, 1]  # x_p (n,)  y_p (n,)
        x_s, y_s = ranges * np.cos(angles), ranges * np.sin(angles)  # x_s (m,)  y_s (m,)
        x_diff = x_s - x_p.reshape(-1, 1)  # x_diff (n, m)
        y_diff = y_s - y_p.reshape(-1, 1)  # y_diff (n, m)
        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)  # dist (n, m)

        return np.any(dist <= car_width)  # True: collision, False: safe

    def calc_cost(self, path, idx_1: int, idx_2: int, parents: dict):
        """ Calculate cost from point_1 to point_2
        """
        # Two kinds of cost:
        #    1. distance between two points
        #    2. change of steering angle
        dist_weight = self.get_parameter('dist_weight').get_parameter_value().double_value
        angle_weight = self.get_parameter('angle_weight').get_parameter_value().double_value

        point_1 = path[idx_1]
        point_2 = path[idx_2]
        dist = np.linalg.norm(point_1 - point_2)

        point_0 = path[parents[idx_1]]
        vec_1 = point_1 - point_0
        vec_2 = point_2 - point_1
        mag_1 = np.linalg.norm(vec_1)
        mag_2 = np.linalg.norm(vec_2)
        angle = np.arccos(vec_1 @ vec_2 / (mag_1 * mag_2))

        cost = dist_weight * dist + angle_weight * angle

        return cost

    def get_steer(self, error):
        """ Get desired steering angle by PID
        """
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error
        steer = kp * error + ki * self.integral + kd * d_error
        return np.clip(steer, -max_control, max_control)

    def get_velocity(self, error):
        """ Get desired velocity based on current error
        """
        err_low_tol = self.get_parameter('err_low_tol').get_parameter_value().double_value
        err_high_tol = self.get_parameter('err_high_tol').get_parameter_value().double_value
        low_vel = self.get_parameter('low_vel').get_parameter_value().double_value
        mid_vel = self.get_parameter('mid_vel').get_parameter_value().double_value
        high_vel = self.get_parameter('high_vel').get_parameter_value().double_value
        max_steer = self.get_parameter('max_control').get_parameter_value().double_value
        atten = self.get_parameter('velocity_attenuation').get_parameter_value().double_value
        
        return (high_vel-low_vel)*np.exp(-abs(error)*atten/err_high_tol)+low_vel
        return high_vel-abs(error)/err_high_tol*(high_vel-low_vel)
        return high_vel-error/max_steer*(high_vel-low_vel)

        if abs(error) <= err_low_tol:
            return high_vel
        elif abs(error) <= err_high_tol:
            return mid_vel
        else:
            return low_vel

    def lidar_callback(self, data: LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        # 1. Read lidar data
        n = len(data.ranges)
        ranges = np.array(data.ranges)
        angles = np.arange(n) * data.angle_increment + data.angle_min

        # 2. Preprocess lidar data
        ranges = self.preprocess_lidar(ranges, data.range_min, data.range_max)

        # 3. Sample candidate points in front of car (polar coordinate, -90° ~ 90°)
        rho_p, theta_p, x_p, y_p, dist = self.sample_nodes(ranges, angles)

        # 4. Calculate potentials for all candidate points
        potentials = self.calc_potential(dist)

        # 5. Get path by choosing best target point at each distance
        path_indices = self.get_path(rho_p, potentials)
        path_x, path_y = x_p[path_indices], y_p[path_indices]
        path_x = np.hstack((0, path_x))
        path_y = np.hstack((0, path_y))

        # 6. Remove redundant waypoints to smooth path
        path_x, path_y = self.path_smoother(path_x, path_y, ranges, angles)

        # 7. Use cubic spline for trajectory planning
        cx, cy, cyaw, ck, s = calc_spline_course(path_x, path_y, ds=0.1)

        # Get speed and steer
        curr_error = cyaw[0]
        steer = self.get_steer(curr_error)
        curr_error = cyaw[-1]
        speed = self.get_velocity(curr_error)

        # Publish Drive message
        self.get_logger().info("Error: %0.2f,\t Steer: %0.2f,\t Vel: %0.2f" % (curr_error * 180.0 / np.pi,
                                                               steer * 180.0 / np.pi,
                                                               speed))
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    print("Potential Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
