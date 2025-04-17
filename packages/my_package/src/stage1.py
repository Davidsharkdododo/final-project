#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from dt_apriltags import Detector
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType


# Define your two PID controllers
class PID:
    def __init__(self, Kp, Ki, Kd, controller_type, negclamp, posclamp):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.controller_type = controller_type.upper()
        self.previous_error = 0
        self.integral = 0
        self.negclamp = negclamp
        self.posclamp = posclamp
        
    def compute(self, error, dt):
        # Proportional term (always included)
        P = self.Kp * error
        
        # Initialize I and D terms to zero
        I = 0
        D = 0
        
        # Add integral term if using PI or PID
        if "I" in self.controller_type:
            self.integral += error * dt
            I = self.Ki * self.integral
        
        # Add derivative term if using PD or PID
        if "D" in self.controller_type:
            D = self.Kd * (error - self.previous_error) / dt
            
        # Update previous error for next iteration
        self.previous_error = error
        
        # The sum of active terms
        correction = P + I + D

        if correction < self.negclamp: return self.negclamp
        elif correction > self.posclamp: return self.posclamp
        else: return correction

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        self.dist_error = None

        
        
        self.enable_lane_following = True
        self.enable_red_detection = True
        self.enable_blue_detection = False
        self.enable_duck_detection = False
        self.enable_duckiebot_detection = True

        self.last_duckiebot_scan_time = rospy.Time(0)




        self.detector = Detector(families="tag36h11")
        self.last_detection_time = rospy.Time(0)
        self.detection_interval = 1  # seconds

        self.base_speed = 0.4
        
        # Lane Control parameters.
        self.lane_controller_type = "P"
        self.lane_Kp = -0.25
        self.lane_Ki = 0.001
        self.lane_Kd = 0.002

        #(Kp, Ki, Kd, controller_type)
        self.lane_pid = PID(-0.22, 0.001, 0.002, "P", -0.4, 0.3)
        self.dist_pid = PID(0.06, 0.001, 0.002, "P", -self.base_speed, 0.2)
        

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
            "red": (np.array([0, 70, 150]), np.array([10, 255, 255])),
            "blue": (np.array([100, 110, 100]), np.array([140, 255, 255])),
            "duck": (np.array([6, 82, 108]),   np.array([22, 255, 255])),
            "duckblue": (np.array([50, 90, 0]),   np.array([110, 255, 255]))
        }
        
        # Flags and counters.
        self.red_line_stopped = False
        self.blue_line_stopped = False
        self.red_line_count = 0  # Count the number of red line detections.
        
        # Dot detection setup.
        self.circlepattern_dims = (7, 2)
        # blob_params = cv2.SimpleBlobDetector_Params()
        # blob_params.minArea = 10
        # blob_params.minDistBetweenBlobs = 2
        # self.simple_blob_detector = cv2.SimpleBlobDetector_create(blob_params)


        # Blob detection setup with improved parameters
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minArea = 5  # Smaller minimum area
        blob_params.maxArea = 500  # Add maximum area constraint
        blob_params.minDistBetweenBlobs = 5
        blob_params.filterByColor = True
        blob_params.blobColor = 0  # Look for dark blobs (adjust if needed)
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.6  # Allow for some deformation
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.8
        blob_params.filterByInertia = True  
        blob_params.minInertiaRatio = 0.5  # Allow less circular blobs
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(blob_params)
        
        # Initialize CvBridge.
        self.bridge = CvBridge()
        
        # Create publishers.
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        self.debug_gray_topic = f"/{self._vehicle_name}/camera_node/image/compressed/gray"
        self.pub_debug_gray = rospy.Publisher(self.debug_gray_topic, Image, queue_size=1)
        
        # Subscribe to the camera feed.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(100)
        
    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred, cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

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
        debug = False

        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges["white"][0], self.hsv_ranges["white"][1])
        
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights.
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        if debug:
            output = image.copy()
            bottom_half_out = output[half_height_start:full_height, :]
        
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        offsets = []
        valid_detections = []
        
        for i, y in enumerate(scan_heights):
            for x in range(center_x, width, 5):
                if white_mask[y, x] > 0:
                    offsets.append(x - center_x)
                    valid_detections.append(i)
                    if debug: cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
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
                    if debug: cv2.circle(bottom_half_out, (x, y), 3, (0, 0, 255), -1)
                    
            if filtered_offsets:
                total_weight = sum([base_weights[i] for i in filtered_detections])
                normalized_weights = [base_weights[i] / total_weight for i in filtered_detections]
                weighted_offsets = [filtered_offsets[i] * normalized_weights[i] for i in range(len(filtered_offsets))]
                error_raw = sum(weighted_offsets)
                error = error_raw * 0.01 - 1.8
            
            if debug:
                color = (0, 0, 255) if len(valid_detections) < 3 else (0, 255, 0)
                cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                debug_img_msg = self.bridge.cv2_to_imgmsg(bottom_half_out, encoding="bgr8")
                debug_img_msg.header.stamp = rospy.Time.now()
                self.pub_lane.publish(debug_img_msg)
        else:
            error = 10
        
        return error

    def detect_red_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["red"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_line_detected = False
        detected_distance = None
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w / 2
                v = y + h
                detected_distance = self.compute_distance_homography(u, v)
                red_line_detected = True
                
        return red_line_detected, detected_distance
    

    def detect_blue_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["blue"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_line_detected = False
        detected_distance = None
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w / 2
                v = y + h
                detected_distance = self.compute_distance_homography(u, v)
                blue_line_detected = True
                
        return blue_line_detected, detected_distance
    
    def detect_duck(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["duck"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        duck_line_detected = False
        detected_distance = None
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w / 2
                v = y + h
                detected_distance = self.compute_distance_homography(u, v)
                duck_line_detected = True
                
        return duck_line_detected, detected_distance
    
    def detect_duckblue(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["duckblue"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        duckblue_detected = False
        dist_error = None
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w / 2
                v = y + h

                # image_height = image.shape[0]  # Get height of the image
                contour_height = 480 - (y + h)  # Bottom of image - bottom 
                distance = 0.0003 * contour_height * contour_height - 0.012 * contour_height + 25.4
                dist_error = distance - 35
                duckblue_detected = True
                
        # rospy.loginfo(distance)
                
        return duckblue_detected, dist_error
        
    def combine_error(self, lane_error, dist_error):
        if dist_error is None: dist_error = 0
        left_speed = self.base_speed - lane_error + dist_error
        right_speed = self.base_speed + lane_error + dist_error
        return left_speed, right_speed



    def publish_cmd(self, left_speed, right_speed):
        # rospy.loginfo(f"left: {left_speed} right: {right_speed}")
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
        self.enable_lane_following = False
        self.enable_red_detection = False
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
        self.enable_lane_following = True
        self.enable_red_detection = True
        self.stop_robot()

    def callback(self, msg):
        
        

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec() if self.last_time else 0.1
        self.last_time = current_time

        # Convert the image.
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed, gray = self.preprocess_image(undistorted)

        # debug_gray_msg = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        # debug_gray_msg.header.stamp = rospy.Time.now()
        # self.pub_debug_gray.publish(debug_gray_msg)


        # run_apriltag_detection = False


        # if (current_time - self.last_detection_time) >= rospy.Duration(self.detection_interval):
        #     run_apriltag_detection = True
        #     self.last_detection_time = current_time

        # detection = None
        # if self.red_line_count == 0 and detection is None and run_apriltag_detection:
        #     detections = self.detector.detect(gray)
        #     if detections:
        #         detection = detections[0]
        #         rospy.loginfo(detection.tag_id)
        




        # --- DOT DETECTION START ---
        duckiebot_timer = False
        if self.enable_duckiebot_detection and (current_time - self.last_duckiebot_scan_time) >= rospy.Duration(0.1): # Seconds
            duckiebot_timer = True
            self.last_duckiebot_scan_time = current_time

        if self.enable_duckiebot_detection and duckiebot_timer:
            # self.dist_error = None
            # Get image dimensions
            height, width = gray.shape
            # Use only the top half of the image
            top_half = gray[0:height//2, :]
            
            found, centers = cv2.findCirclesGrid(
                top_half,  # Use only top half of image
                patternSize=self.circlepattern_dims,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=self.simple_blob_detector
            )
            if found and centers is not None and centers.shape[0] >= 2:
                # Adjust y-coordinates to account for ROI
                # centers[:,:,1] are the y-coordinates
                
                # Using dots at indices 5 and 6 as an example.
                dot0 = centers[5, 0]
                dot1 = centers[6, 0]
                distance_dots = np.linalg.norm(dot0 - dot1)
                dist = 1.52 * distance_dots ** (-0.5) # fitted equation https://www.desmos.com/calculator/wclkp3m3es
                # self.dist_error = dist - 0.4 # attempt to follow at 40cm distance
                # rospy.loginfo(f"Dot detection: distance {distance_dots} pixels, {distance}m")

        duckblue_detected, self.dist_error = self.detect_duckblue(undistorted)


        # Detect red line.
        if self.enable_red_detection:
            red_detected, red_distance = self.detect_red_line(preprocessed)
            if red_detected and red_distance is not None and red_distance < 0.15:
                if not self.red_line_stopped:
                    self.red_line_count += 1
                    rospy.loginfo("Red line detected, count: %d", self.red_line_count)
                    if self.red_line_count == 3:
                        rospy.loginfo("Third red line detected. Executing left 90 degree turn with 50cm radius.")
                        self.stop_robot()
                        rospy.sleep(0.5)
                        self.perform_left_turn()
                    elif self.red_line_count == 4:
                        rospy.loginfo("Fourth red line detected. Thinking")
                        self.stop_robot()
                        detections = self.detector.detect(gray)
                        detection = detections[-1]
                        rospy.sleep(0.5)
                        if not detection: rospy.loginfo("not detection")
                        if not detection.tag_id == 133: rospy.loginfo("not detection.tag_id == 133")
                        if detection.tag_id == 133:
                            rospy.loginfo("Left turn")
                            self.perform_left_turn()
                    else:
                        rospy.loginfo("Red line detected within 20cm. Stopping for 0.5 sec.")
                        self.stop_robot()
                        rospy.sleep(0.5)
                    self.red_line_stopped = True
            else:
                self.red_line_stopped = False




        # if self.red_line_count == 0:
        #     blue_detected, blue_distance = self.detect_blue_line(preprocessed)
        #     if blue_detected and blue_distance < 0.15:
        #         duck_detected, duck_distance = self.detect_duck(preprocessed)
        #         if duck_detected and duck_distance < 0.18:
        #             self.stop_robot()
        #         else:
        #             self.stop_robot()
        #             rospy.sleep(0.5)





                
        
        
        # Continue with lane detection and control.
        if self.enable_lane_following:
            lane_error = self.detect_lane_fast(preprocessed)
            dist_corr = None
            if self.dist_error: dist_corr = self.dist_pid.compute(self.dist_error, dt)
            if lane_error:
                lane_corr = self.lane_pid.compute(lane_error, dt)
                left_speed, right_speed = self.combine_error(lane_corr, dist_corr)
                self.publish_cmd(left_speed, right_speed)

            

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()