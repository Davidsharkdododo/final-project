#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped

# Assuming DTROS and NodeType are defined elsewhere in your project (e.g., from Duckietown)
from duckietown.dtros import DTROS, NodeType

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Get control parameters from ROS parameters.
        self.controller_type = "P"
        self.Kp = -0.25
        self.Ki = 0.025
        self.Kd = 0.05
        self.base_speed = 0.4

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters.s
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
        
        # Color detection parameters (HSV ranges).
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            # "white": (np.array([0, 0, 216]), np.array([179, 55, 255]))
            "white": (np.array([0, 0, 216*0.85]), np.array([179*1.1, 55*1.2, 255]))
        }
        
        # Initialize CvBridge.
        self.bridge = CvBridge()
        
        # Create publishers.
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        
        # Now that all variables are initialized, subscribe to the camera feed.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(2)
        

    def undistort_image(self, image):
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred


    def detect_lane_fast(self, image):
        """
        Optimized fast lane detection using fixed-height scanning.
        Scans for white pixels at fixed heights in the bottom half of the image,
        checking only every third pixel for speed.
        """
        # Convert to HSV and get white mask - only process bottom half
        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        # Convert to HSV and create mask only for bottom half
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges["white"][0], self.hsv_ranges["white"][1])
        
        # Get dimensions and center
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        
        # Create a simple output image (minimal drawing for speed)
        output = image.copy()
        bottom_half_out = output[half_height_start:full_height, :]
        
        # Base weights (will be redistributed if points are missing)
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        offsets = []
        valid_detections = []
        
        # Loop through each scan height
        for i, y in enumerate(scan_heights):
            # Scan from center to right edge every 3 pixels for speed
            for x in range(center_x, width, 3):
                # If we found a white pixel
                if white_mask[y, x] > 0:
                    offset = x - center_x
                    offsets.append(offset)
                    valid_detections.append(i)
                    
                    # Minimal visualization - just circles at detection points
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
                    break
        
        # Calculate error - redistribute weights for missing points
        error = None
        if offsets:
            # Find the bottom-most detection (should be the most reliable)
            bottom_idx = valid_detections.index(max([i for i in valid_detections]))
            bottom_offset = offsets[bottom_idx]
            
            # Filter out detections that are too far from the bottom detection
            # (likely background noise or other lane markers)
            max_deviation = 10  # maximum allowed pixel deviation from bottom point
            
            filtered_offsets = []
            filtered_detections = []
            
            for i, offset in enumerate(offsets):
                if offset - bottom_offset <= max_deviation:
                    filtered_offsets.append(offset)
                    filtered_detections.append(valid_detections[i])
                else:
                    # Draw rejected points in red
                    detection_idx = valid_detections[i]
                    y = scan_heights[detection_idx]
                    x = center_x + offset
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 0, 255), -1)
            
            # Use the filtered detections
            if filtered_offsets:
                # Create new weights based on which points were detected
                total_weight = sum([base_weights[i] for i in filtered_detections])
                normalized_weights = [base_weights[i] / total_weight for i in filtered_detections]
                
                # Calculate weighted sum of offsets
                weighted_offsets = [filtered_offsets[i] * normalized_weights[i] for i in range(len(filtered_offsets))]
                error_raw = sum(weighted_offsets)
                
                # Scale the error
                error = error_raw * 0.01 -1.2
            
            # Minimal text display
            if len(valid_detections) < 3:
                # If we have very few points, make text red as a warning
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                
            cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output, error


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
        # Differential drive: adjust left/right speeds based on control output.
        if control_output >0.3: control_output = min(control_output, 0.3) # Scale control output appropriately.
        if control_output <-0.3: control_output = max(control_output, -0.3) # Scale control output appropriately.
        left_speed = self.base_speed - control_output
        right_speed = self.base_speed + control_output
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
    
    def callback(self, msg):
        current_time = rospy.Time.now()
        if self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
        else:
            dt = 0.1  # default dt value for the first callback
        self.last_time = current_time
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        output, error = self.detect_lane_fast(preprocessed)
        lane_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        self.pub_lane.publish(lane_msg)


        if error is not None:
            # rospy.loginfo("Control error: %.3f", error)
            control_output = self.get_control_output(error, dt)
            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
