#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from dt_apriltags import Detector
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import Header, ColorRGBA
from duckietown.dtros import DTROS, NodeType

class AprilTagLaneController(DTROS):
    def __init__(self, node_name):
        super(AprilTagLaneController, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Lane following control parameters
        self.controller_type = "P"
        self.Kp = -0.25
        self.Ki = 0.025
        self.Kd = 0.05
        self.base_speed = 0.4

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        
        # Homography matrix (maps image coordinates to ground coordinates)
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # HSV ranges for lane and red line detection
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, int(216*0.85)]), np.array([int(179*1.1), int(55*1.2), 255])),
            "red": (np.array([0, 70, 150]), np.array([10, 255, 255]))
        }
        
        # AprilTag detection parameters
        self.detector = Detector(families="tag36h11")
        self.last_detection_time = rospy.Time(0)
        self.detection_interval = 0.33  # seconds
        
        # Mapping: tag ID 21 is red (Stop Sign, 3 sec),
        #          tag ID 133 is blue (T-Intersection, 2 sec),
        #          tag ID 94 is green (UofA Tag, 1 sec).
        self.tag_mapping = {
            21: {"color": "red", "stop_duration": 3.0},
            22: {"color": "red", "stop_duration": 3.0},
            133: {"color": "blue", "stop_duration": 2.0},
            93: {"color": "green", "stop_duration": 1.0},
            94: {"color": "green", "stop_duration": 1.0}
        }
        # Default mapping if tag ID is not recognized
        self.default_tag = {"color": "white", "stop_duration": 0.5}
        
        # Variables to store tag and red line detection states
        self.last_tag_info = None
        self.tag_visible = False
        self.tag_lost_time = None
        self.red_line_detected = False
        self.at_intersection = False
        self.lane_following_enabled = True
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self._led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        
        # Image publishers for debug visualization
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=1)
        self.apriltag_topic = f"/{self._vehicle_name}/camera_node/image/compressed/apriltag"
        self.pub_apriltag = rospy.Publisher(self.apriltag_topic, Image, queue_size=1)
        
        # Subscribe to camera feed
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(2)
        self.log("AprilTag Lane Controller initialized.")

    def undistort_image(self, image):
        """Undistort the image using camera calibration parameters."""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        """Preprocess image for lane detection."""
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        # Split into RGB channels for different detections
        return blurred, cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    def compute_distance_homography(self, u, v):
        """Compute distance to a point using homography."""
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def detect_lane_fast(self, image):
        """Fast lane detection using fixed-height scanning."""
        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        # Use white lane for now
        lane_color = "white"
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        lane_mask = cv2.inRange(hsv, self.hsv_ranges[lane_color][0], self.hsv_ranges[lane_color][1])
        
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        output = image.copy()
        bottom_half_out = output[half_height_start:full_height, :]
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        offsets = []
        valid_detections = []
        
        for i, y in enumerate(scan_heights):
            for x in range(center_x, width, 3):
                if lane_mask[y, x] > 0:
                    offsets.append(x - center_x)
                    valid_detections.append(i)
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
                    break
        
        error = None
        if offsets:
            bottom_idx = valid_detections.index(max(valid_detections))
            bottom_offset = offsets[bottom_idx]
            max_deviation = 10
            filtered_offsets = []
            filtered_detections = []
            for i, offset in enumerate(offsets):
                if offset - bottom_offset <= max_deviation:
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
                error = error_raw * 0.01 - 1.5
                rospy.loginfo(error)
            color = (0, 0, 255) if len(valid_detections) < 3 else (0, 255, 0)
            cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output, error

    def detect_red_line(self, image):
        """Detect red lines in the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_distance = None
        red_line_detected = False
        annotated = image.copy()
        
        lower, upper = self.hsv_ranges["red"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                u = x + w / 2
                v = y + h
                distance = self.compute_distance_homography(u, v)
                cv2.putText(annotated, f"Red line: {distance:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                red_line_detected = True
                detected_distance = distance
                
        return red_line_detected, detected_distance, annotated

    def update_leds(self, color_str):
        """Update LEDs based on the provided color string."""
        num_leds = 5
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()
        
        if color_str == "red":
            color = ColorRGBA(r=1, g=0, b=0, a=1)
        elif color_str == "blue":
            color = ColorRGBA(r=0, g=0, b=1, a=1)
        elif color_str == "green":
            color = ColorRGBA(r=0, g=1, b=0, a=1)
        else:  # "white" or unknown
            color = ColorRGBA(r=1, g=1, b=1, a=1)
            
        leds = [color for _ in range(num_leds)]
        pattern_msg.rgb_vals = leds
        self._led_pub.publish(pattern_msg)

    def calculate_p_control(self, error, dt):
        return self.Kp * error

    def calculate_pd_control(self, error, dt):
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
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
        control_output = max(min(control_output, 0.3), -0.3)
        left_speed = self.base_speed - control_output
        right_speed = self.base_speed + control_output
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
        
    def stop_robot(self):
        """Stop the robot by publishing zero velocity commands."""
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = 0
        cmd_msg.vel_right = 0
        self._publisher.publish(cmd_msg)
    
    def callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec() if self.last_time else 0.1
        self.last_time = current_time
        
        # Convert the image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed, gray = self.preprocess_image(undistorted)
        
        # Detect red line
        red_line_detected, red_line_distance, red_line_image = self.detect_red_line(preprocessed)
        
        # Update red line detection state
        if red_line_detected and red_line_distance < 0.20:  # If within 20cm
            self.red_line_detected = True
            rospy.loginfo("RED LINE DETECTED at distance %.2f m", red_line_distance)
        else:
            self.red_line_detected = False
        
        # Detect AprilTags (rate-limited)
        run_apriltag_detection = False
        if (current_time - self.last_detection_time) >= rospy.Duration(self.detection_interval):
            run_apriltag_detection = True
            self.last_detection_time = current_time
        
        # Run AprilTag detection if needed
        if run_apriltag_detection:
            detections = self.detector.detect(gray, estimate_tag_pose=False)
            
            if detections:
                # Process the first detected tag
                detection = detections[0]
                corners = detection.corners.astype(int)
                pts = corners.reshape((-1, 1, 2))
                cv2.polylines(red_line_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                center = detection.center.astype(int)
                cv2.putText(red_line_image, f"ID: {detection.tag_id}", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                tag_info = self.tag_mapping.get(detection.tag_id, self.default_tag)
                self.update_leds(tag_info["color"])
                
                # Remember the tag info and reset loss timer
                self.last_tag_info = tag_info
                self.tag_visible = True
                self.tag_lost_time = None
                rospy.loginfo("Detected AprilTag ID: %d (%s)", detection.tag_id, tag_info["color"])
            else:
                self.tag_visible = False
                # If a previous tag exists, keep its color
                if self.last_tag_info is not None:
                    self.update_leds(self.last_tag_info["color"])
                else:
                    self.update_leds("white")
                    
                # NOTE: We still track when the tag was lost, but we DON'T use it to trigger stopping
                if self.tag_lost_time is None:
                    self.tag_lost_time = current_time
        
        # Publish debug images
        lane_result, lane_error = self.detect_lane_fast(preprocessed)
        lane_msg = self.bridge.cv2_to_imgmsg(lane_result, encoding="bgr8")
        self.pub_lane.publish(lane_msg)
        
        combined_image = red_line_image.copy()
        red_line_msg = self.bridge.cv2_to_imgmsg(combined_image, encoding="bgr8")
        self.pub_apriltag.publish(red_line_msg)
        
        # Decision logic for lane following vs. stopping at intersections
        # --------------------------------------------------------------------------------
        # ONLY stop when we detect a red line with tag instructions
        if self.red_line_detected and not self.at_intersection:
            # Disable lane following
            self.lane_following_enabled = False
            
        # Use tag info if available, otherwise use default behavior
            if self.last_tag_info is not None:
                # We have tag instructions
                duration = self.last_tag_info["stop_duration"]
                color = self.last_tag_info["color"]
                rospy.loginfo("STOPPING at red line for %.1f seconds based on %s tag", duration, color)
            else:
                # No tag seen, use default behavior
                duration = 0.5  # 0.5 seconds
                color = "white"
                rospy.loginfo("STOPPING at red line for %.1f seconds (default behavior, no tag seen)", duration)
                # Set white LEDs for default behavior
                self.update_leds(color)
            
            # Mark that we're at an intersection to prevent multiple stops
            self.at_intersection = True
            
            # Stop the robot
            self.stop_robot()
            rospy.sleep(duration)
            
            # Resume driving (will be handled in next callback)
            self.lane_following_enabled = True
            
        # Reset intersection flag once we've passed the line
        elif not self.red_line_detected and self.at_intersection:
            rospy.loginfo("Passed intersection, resetting flags")
            self.at_intersection = False
            # Only clear tag info after we've passed the intersection
            # This allows us to properly handle consecutive intersections
            self.last_tag_info = None
                
        # Normal lane following when enabled
        elif self.lane_following_enabled and lane_error is not None:
            control_output = self.get_control_output(lane_error, dt)
            self.publish_cmd(control_output)
            
    def run(self):
        """Main loop that keeps the node alive."""
        rospy.spin()

if __name__ == '__main__':
    node = AprilTagLaneController(node_name='apriltag_lane_controller')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass