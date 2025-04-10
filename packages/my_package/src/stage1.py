#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Control parameters.
        self.controller_type = "P"
        self.Kp = -0.25
        self.Ki = 0.001
        self.Kd = 0.002
        self.base_speed = 0.4

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        
        # Homography matrix (maps image coordinates to ground coordinates).
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # HSV ranges for lane and red line detection.
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, int(216 * 0.85)]), np.array([int(179 * 1.1), int(55 * 1.2), 255])),
            "red": (np.array([0, 70, 150]), np.array([10, 255, 255]))
        }
        
        # Flags and counters.
        self.red_line_stopped = False
        self.red_line_count = 0  # Count the number of red line detections.
        
        # Dot detection setup.
        self.circlepattern_dims = (7, 3)
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minArea = 10
        blob_params.minDistBetweenBlobs = 2
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(blob_params)
        
        # Initialize CvBridge.
        self.bridge = CvBridge()
        
        # Create publishers.
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        
        # Subscribe to the camera feed.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(2)
        
    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def compute_distance_homography(self, u, v):
        """Compute distance from image point (u, v) using homography."""
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def detect_lane_fast(self, image):
        """
        Fast lane detection using fixed-height scanning.
        Scans for white pixels in the bottom half of the image.
        """
        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges["white"][0], self.hsv_ranges["white"][1])
        
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights.
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        output = image.copy()
        bottom_half_out = output[half_height_start:full_height, :]
        
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        offsets = []
        valid_detections = []
        
        for i, y in enumerate(scan_heights):
            for x in range(center_x, width, 3):
                if white_mask[y, x] > 0:
                    offsets.append(x - center_x)
                    valid_detections.append(i)
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
                    break
        
        error = None
        if offsets:
            bottom_idx = valid_detections.index(max(valid_detections))
            bottom_offset = offsets[bottom_idx]
            max_deviation = 10  # allowed pixel deviation
            
            filtered_offsets = []
            filtered_detections = []
            for i, offset in enumerate(offsets):
                if abs(offset - bottom_offset) <= max_deviation:
                    filtered_offsets.append(offset)
                    filtered_detections.append(valid_detections[i])
                else:
                    detection_idx = valid_detections[i]
                    y = scan_heights[detection_idx]
                    x = center_x + offset
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 0, 255), -1)
                    
            if filtered_offsets:
                total_weight = sum([base_weights[i] for i in filtered_detections])
                normalized_weights = [base_weights[i] / total_weight for i in filtered_detections]
                weighted_offsets = [filtered_offsets[i] * normalized_weights[i] for i in range(len(filtered_offsets))]
                error_raw = sum(weighted_offsets)
                error = error_raw * 0.01 - 1.3
            
            color = (0, 0, 255) if len(valid_detections) < 3 else (0, 255, 0)
            cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output, error

    def detect_red_line(self, image):
        """
        Detect red lines in the image.
        Returns a tuple: (red_line_detected, distance, annotated_image).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["red"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_line_detected = False
        detected_distance = None
        annotated = image.copy()
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:  # Minimum area threshold.
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Compute the center-bottom of the bounding box.
                u = x + w / 2
                v = y + h
                detected_distance = self.compute_distance_homography(u, v)
                cv2.putText(annotated, f"Red line: {detected_distance:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                red_line_detected = True
                
        return red_line_detected, detected_distance, annotated

    def calculate_p_control(self, error, dt):
        return self.Kp * error

    def calculate_pd_control(self, error, dt):
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        rospy.loginfo(derivative)
        output = self.Kp * error + self.Kd * derivative
        self.prev_error = error
        return output

    def calculate_pid_control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def get_control_output(self, error, dt):
        ctrl_type = self.controller_type.upper()
        if ctrl_type == "P":
            return self.calculate_p_control(error, dt)
        elif ctrl_type == "PD":
            return self.calculate_pd_control(error, dt)
        elif ctrl_type == "PID":
            return self.calculate_pid_control(error, dt)
        else:
            rospy.logwarn("Unknown controller type '%s'. Using P controller.", self.controller_type)
            return self.calculate_p_control(error, dt)
        
    def publish_cmd(self, control_output):
        # Limit control output.
        if control_output > 0.3: 
            control_output = 0.3
        if control_output < -0.3: 
            control_output = -0.3
        left_speed = self.base_speed - control_output
        right_speed = self.base_speed + control_output
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
        
    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = 0.0
        cmd_msg.vel_right = 0.0
        self._publisher.publish(cmd_msg)
        
    def perform_left_turn(self):
        """
        Execute a left turn of 90 degrees with a turning radius of 50 cm.
        For a turning radius R = 0.5 m and a chosen linear speed (e.g. 0.3 m/s),
        the angular velocity omega = 0.3 / 0.5 = 0.6 rad/s.
        A 90° turn requires an angle of pi/2 ≈ 1.57 rad, so the turn duration is:
              t = 1.57 / 0.6 ≈ 2.62 seconds.
        """
        turn_duration = 4  # seconds
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(10)  # 10 Hz
        rospy.loginfo("Turning left 90 degrees with 50cm radius; duration: %.2f sec", turn_duration)
        while rospy.Time.now().to_sec() - start_time < turn_duration and not rospy.is_shutdown():
            cmd_msg = WheelsCmdStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            # Differential wheel speeds for a left turn.
            # Example values: left wheel slower than right wheel.
            cmd_msg.vel_left = 0.41
            cmd_msg.vel_right = 0.7
            self._publisher.publish(cmd_msg)
            rate.sleep()
        self.stop_robot()

    def callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec() if self.last_time else 0.1
        self.last_time = current_time

        # Convert the image.
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        
        # --- DOT DETECTION START ---
        found, centers = cv2.findCirclesGrid(
            undistorted, 
            patternSize=self.circlepattern_dims, 
            flags=cv2.CALIB_CB_SYMMETRIC_GRID, 
            blobDetector=self.simple_blob_detector
        )
        if found and centers is not None and centers.shape[0] >= 2:
            # Using dots at indices 5 and 6 as an example.
            dot0 = centers[5, 0]
            dot1 = centers[6, 0]
            distance_dots = np.linalg.norm(dot0 - dot1)
            if distance_dots > 9:
                rospy.loginfo("Dot detection: distance %.2f pixels > 10, stopping robot." % distance_dots)
                self.stop_robot()
        
        # Detect red line.
        red_detected, red_distance, red_annotated = self.detect_red_line(preprocessed)
        if red_detected and red_distance is not None and red_distance < 0.15:
            if not self.red_line_stopped:
                self.red_line_count += 1
                rospy.loginfo("Red line detected, count: %d", self.red_line_count)
                if self.red_line_count == 3:
                    rospy.loginfo("Third red line detected. Executing left 90 degree turn with 50cm radius.")
                    self.stop_robot()
                    rospy.sleep(0.5)
                    self.perform_left_turn()
                else:
                    rospy.loginfo("Red line detected within 20cm. Stopping for 0.5 sec.")
                    self.stop_robot()
                    rospy.sleep(0.5)
                self.red_line_stopped = True
        else:
            self.red_line_stopped = False
        
        # Continue with lane detection and control.
        output, error = self.detect_lane_fast(preprocessed)

        if error is not None:
            control_output = self.get_control_output(error, dt)
            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()