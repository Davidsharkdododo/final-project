#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from dt_apriltags import Detector
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Header, ColorRGBA
import message_filters


Wheel_rad = 0.0318
Wheel_base = 0.1

def compute_distance(ticks):
    rotations = ticks/135
    return 2 * 3.1415 * Wheel_rad * rotations

def compute_ticks(distance):
    rotations = distance / (2 * 3.1415 * Wheel_rad)
    return rotations * 135

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
        
    def compute(self, error, dt=None):
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

        if correction < self.negclamp:
            return self.negclamp
        elif correction > self.posclamp:
            return self.posclamp
        else:
            return correction


class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        

        self.dist_error = None
        self.first_line_do_turn = None  # track first red line decision
        self.fourth_line_do_turn = False
        self.fourth_line_do_turn = False

        self.turn_locked_on = False
        self.left_turn_start_time = None
        self.dynamic_control_start_time = None
        self.dynamic_control_finished = False
        self.aligning_stopped = False
        self.park_start_time = None

        self._ticks_left = 0
        self._ticks_right = 0
        self.init_ticks_left = 0
        self.init_ticks_right = 0

        
        self.enable_lane_following = True
        self.enable_red_detection = True
        self.enable_blue_detection = False
        self.enable_duck_detection = False
        self.enable_duckblue_detection = True
        self.waiting_for_ducks = False
        self.enable_apriltag_detection = True
        self.blue_line_count      = 0
        self.last_blue_stop_time  = rospy.Time(0)
        self.blue_cooldown        = rospy.Duration(10)   # 3 s ignore between blue detections
        self.enable_blue_phase    = False
        self.tagid = None
        self.blue_line_ref_ticks = None
        self.led_color = "blue"

        self.last_duckiebot_scan_time = rospy.Time(0)
        
        self.detector = Detector(families="tag36h11")
        self.last_detection_time = rospy.Time(0)
        self.detection_interval = 1  # seconds

        self.base_speed = 0.5
        
        # PID controllers
        self.lane_pid = PID(-0.22, 0.01, 0.01, "P", -0.2, 0.3)
        self.lane_pid_centroid = PID(-0.01, 0.01, 0.01, "P", -0.4, 0.5)
        self.lane_pid_centroid2 = PID(-0.12, 0.01, 0.01, "P", -0.4, 0.4)
        self.dist_pid = PID(0.06, 0.001, 0.002, "P", -self.base_speed, 0.2)
        self.tick_pid = PID(0.02, 0.001, 0.002, "P", -0.3, 0.3)
        self.dynamic_pid = PID(0.055, 0.001, 0.002, "P", -0.3, 0.3)
        self.rotation_pid = PID(0.03, 0.008, 0.01, "PID", -4000, 4000)
        self.tagpark_pid = PID(-0.06, 0.001, 0.003, "P", -0.2, 0.2)

        self.red_line_count = 5
        self.movements = 0
        self.park = 3

        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        
        # Homography matrix
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        self.red_line_stopped = False
        self.blue_line_stopped = False
        # track when we last stopped for a red line
        self.last_red_stop_time = rospy.Time(0)
        self.red_cooldown = rospy.Duration(3)   # 5 seconds of “ignore” time
        self.use_yellow_lane = False

        # Variables to manage lane switching/dot detection cooldown.
        self.lane_switch_start_time = None   # Set when lane switch is triggered.
        self.lane_switch_cooldown = rospy.Duration(10)
        # Create a blob detector for duckiebot (circle grid) detection.
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minArea = 10
        blob_params.minDistBetweenBlobs = 2
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(blob_params)
        # Expected circle grid dimensions for the duckiebot pattern.
        self.circlepattern_dims = [7, 3]

        # HSV ranges
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, int(216 * 0.85)]), np.array([int(179 * 1.1), int(55 * 1.2), 255])),
            "red": (np.array([0, 70, 150]), np.array([10, 255, 255])),
            "red2": (np.array([0, 85, 202]), np.array([179, 255, 255])),
            "blue": (np.array([100, 110, 100]), np.array([140, 255, 255])),
            "duck": (np.array([6, 82, 108]),   np.array([22, 255, 255])),
            "duckblue": (np.array([50, 90, 0]),   np.array([110, 255, 255])),
            "parkblue": (np.array([60, 39, 96]),   np.array([107, 80, 204]))
        }
        
        # CvBridge and publishers/subscribers
        self.bridge = CvBridge()
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        self.debug_gray_topic = f"/{self._vehicle_name}/camera_node/image/compressed/gray"
        self.pub_debug_gray = rospy.Publisher(self.debug_gray_topic, Image, queue_size=1)
        self._led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        # self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        # self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        
        image_sub = message_filters.Subscriber(self._camera_topic, CompressedImage)
        left_sub = message_filters.Subscriber(self._left_encoder_topic, WheelEncoderStamped)
        right_sub = message_filters.Subscriber(self._right_encoder_topic, WheelEncoderStamped)

        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, left_sub, right_sub], 
            queue_size=1, 
            slop=0.1  # 100ms tolerance
        )
        ts.registerCallback(self.callback)


        self.rate = rospy.Rate(100)
        self.encoder_rate_slow = rospy.Rate(1)
        self.encoder_rate_fast = rospy.Rate(5)



    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred, cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    def compute_distance_homography(self, u, v):
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def detect_duckblue(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["duckblue"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        duckblue_detected = False
        dist_error = None
        left_dist = None
        right_dist = None
        img_width = image.shape[1]

        valid = [c for c in contours if cv2.contourArea(c) > 500]
        if valid:
            rects = [cv2.boundingRect(c) for c in valid]
            xs = [x for x, y, w, h in rects]
            rights = [x + w for x, y, w, h in rects]
            leftmost = min(xs)
            rightmost = max(rights)
            left_dist = leftmost
            right_dist = img_width - rightmost

            c = max(valid, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            contour_height = 480 - (y + h)
            distance = 0.0003 * contour_height**2 - 0.012 * contour_height + 25.4
            dist_error = distance - 35
            duckblue_detected = True

        return duckblue_detected, dist_error, left_dist, right_dist

    def detect_lane_fast(self, image):
        scaled_image = cv2.resize(image, (160, 120))
        bottom_right = scaled_image[80:, 60:]
        
        hsv = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2HSV)
        
        # Create mask for color
        lane_color = "yellow" if self.use_yellow_lane else "white"
        mask = cv2.inRange(hsv, self.hsv_ranges[lane_color][0], self.hsv_ranges[lane_color][1])
        
        # Find contours directly on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process results
        if not contours: 
            # rospy.loginfo("not contours")
            return 1000
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50: 
            # rospy.loginfo("cv2.contourArea(largest_contour) < 50:")
            return 1000
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            # rospy.loginfo("M['m00'] == 0")
            return 1000
        cx = int(M['m10'] / M['m00'])

        # rospy.loginfo(f"centroid: {cx} | M20x: {M['m20']} | M02y: {M['m02']}")
        # rospy.loginfo(f"centroid: {cx}")

        error = cx - 80

        return error
    
    def detect_lane_fast_wide(self, image, side):
        scaled_image = cv2.resize(image, (160, 120))
        if side == "left":
            bottom_left = scaled_image[80:, :60]
            hsv = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2HSV)
        elif side == "right":
            bottom_right = scaled_image[80:, 60:]
            hsv = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2HSV)
    
        
        # Create mask for color
        lane_color = "yellow" if self.use_yellow_lane else "white"
        mask = cv2.inRange(hsv, self.hsv_ranges[lane_color][0], self.hsv_ranges[lane_color][1])
        
        # Find contours directly on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process results
        if not contours: 
            # rospy.loginfo("not contours")
            return 1000
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50: 
            # rospy.loginfo("cv2.contourArea(largest_contour) < 50:")
            return 1000
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            # rospy.loginfo("M['m00'] == 0")
            return 1000
        cx = int(M['m10'] / M['m00'])

        # rospy.loginfo(f"centroid: {cx} | M20x: {M['m20']} | M02y: {M['m02']}")
        # rospy.loginfo(f"centroid: {cx}")
        if side == "left":
            error = cx - 10
        elif side == "right":
            error = cx - 75

        # rospy.loginfo(f"error: {error}")

        return error


    def detect_red_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["red"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w/2
                v = y + h
                dist = self.compute_distance_homography(u, v)
                return True, dist
        return False, None
    
    def detect_red_line_angle(self, image):
        """
        Detect the angle of a red line by shooting rays down from above the contour
        and measuring the difference in y-values where they first hit the contour.
        """
        # Convert to HSV and create mask for red
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["red2"]
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None  # No red object detected
            
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None  # Too small to be reliable
        
        # Find the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Define ray starting points (15px left and right of center)
        left_x = cx - 15
        right_x = cx + 15
        
        # Make sure ray starting points are within image bounds
        left_x = max(0, min(left_x, mask.shape[1]-1))
        right_x = max(0, min(right_x, mask.shape[1]-1))
        
        # Start rays from top of the image (or some distance above contour)
        start_y = max(0, cy - 50)  # Start 50px above contour center
        
        # Shoot rays downward
        left_y = None
        for y in range(start_y, mask.shape[0]):
            if mask[y, left_x] > 0:  # Found red pixel
                left_y = y
                break
                
        right_y = None
        for y in range(start_y, mask.shape[0]):
            if mask[y, right_x] > 0:  # Found red pixel
                right_y = y
                break
        
        # Calculate angle from y-difference
        if left_y is not None and right_y is not None:
            # Positive angle means right side is higher (red line sloping up to the right)
            y_diff = left_y - right_y
            # Convert y_diff to angle (approximation)
            # tan(angle) ≈ y_diff / x_diff, where x_diff is 40 pixels
            angle = np.degrees(np.arctan2(y_diff, 30))
            
            # rospy.loginfo(f"Red line angle: {angle:.2f}° (y_diff: {y_diff})")
            return angle
        else:
            return None
        
    def align_with_red_line(self, image, dt):
        """
        Rotates the Duckiebot to align with a red line (make the line horizontal).
        Uses the detect_red_line_angle function to get the current angle and 
        applies differential steering to rotate until the angle is close to 0.
        """
        self.enable_lane_following = False
        self.enable_red_detection = False
        
        # Detect the current angle of the red line
        angle = self.detect_red_line_angle(image)
        
        if angle is None:
            # No red line detected, stop and return
            self.stop_robot()
            return False
        
        # Check if we're already aligned (within threshold)
        if abs(angle) < 1:  # 1 degree threshold for alignment
            rospy.loginfo("Red line aligned (angle: %.2f)", angle)
            self.stop_robot()
            self.aligning_stopped = True
            self.movements += 1
        
        # Calculate rotation correction using PID controller
        # Note: You'll need to add your own PID controller for this (e.g., self.rotation_pid)
        rotation_correction = self.rotation_pid.compute(angle, dt)
        
        # Apply the rotation_correction to refine the rotation speed

        left_speed = 0
        right_speed = rotation_correction


        
        # Publish wheel commands
        rospy.loginfo(f"Left: {left_speed}, Right: {right_speed}")

        self.publish_cmd(left_speed, right_speed)
    
    def detect_blue_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["blue"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w/2
                v = y + h
                dist = self.compute_distance_homography(u, v)
                return True, dist
        return False, None
    
    def detect_duck(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = self.hsv_ranges["duck"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w/2
                v = y + h
                dist = self.compute_distance_homography(u, v)
                return True, dist
        return False, None

    def combine_error(self, lane_error, dist_error):
        if dist_error is None: dist_error = 0
        left_speed = self.base_speed - lane_error + dist_error
        right_speed = self.base_speed + lane_error + dist_error
        return left_speed, right_speed

    def publish_cmd(self, left_speed, right_speed):
        # rospy.loginfo(f"Left: {left_speed}, Right: {right_speed}")
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self._publisher.publish(cmd)

    def stop_robot(self):
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.vel_left = 0.0
        cmd.vel_right = 0.0
        # rospy.loginfo("stop_robot")
        self._publisher.publish(cmd)

    def perform_left_turn(self, image, radius = None, time = 3):
        self.enable_lane_following = False
        self.enable_red_detection = False
        if self.left_turn_start_time is None: 
            self.left_turn_start_time = rospy.Time.now().to_sec()
            self.init_ticks_left = self._ticks_left
            self.init_ticks_right = self._ticks_right
        left_dist = self._ticks_left - self.init_ticks_left
        right_dist = self._ticks_right - self.init_ticks_right

        if radius is not None:
            dist_ratio = (radius - 0.05) / (radius + 0.05)
            
            diffr = abs(left_dist) - abs(right_dist)*dist_ratio

            corr = self.tick_pid.compute(diffr)

            # rospy.loginfo(f"diff: {diff} corr: {corr}")

            self.publish_cmd((0.5 * (radius - 0.05) / radius) - corr, (0.5 * (radius + 0.05) / radius) + corr)
        else:
            diffs = abs(left_dist) - abs(right_dist)
            corrs = self.tick_pid.compute(diffs)

            self.publish_cmd(0.4 - corrs, 0.4 + corrs)

        lane_error = self.detect_lane_fast(image)
        # rospy.loginfo(lane_error)
        if abs(lane_error) < 20 and rospy.Time.now().to_sec() - self.left_turn_start_time > time:
            self.enable_lane_following = True
            self.enable_red_detection = True
            self.turn_locked_on = True
            rospy.loginfo("locked on")

    def perform_left_turn_park(self, image, side, radius = None, time = 3):
        self.enable_lane_following = False
        self.enable_red_detection = False
        if self.left_turn_start_time is None: 
            self.left_turn_start_time = rospy.Time.now().to_sec()
            self.init_ticks_left = self._ticks_left
            self.init_ticks_right = self._ticks_right
        left_dist = self._ticks_left - self.init_ticks_left
        right_dist = self._ticks_right - self.init_ticks_right

        if radius is not None:
            dist_ratio = (radius - 0.05) / (radius + 0.05)
            
            diffr = abs(left_dist) - abs(right_dist)*dist_ratio

            corr = self.tick_pid.compute(diffr)

            # rospy.loginfo(f"diff: {diff} corr: {corr}")

            self.publish_cmd((0.5 * (radius - 0.05) / radius) - corr, (0.5 * (radius + 0.05) / radius) + corr)
        else:
            diffs = abs(left_dist) - abs(right_dist)
            corrs = self.tick_pid.compute(diffs)

            self.publish_cmd(0.4 - corrs, 0.4 + corrs)

        lane_error = self.detect_lane_fast_wide(image, side)
        # rospy.loginfo(lane_error)
        if abs(lane_error) < 10 and rospy.Time.now().to_sec() - self.left_turn_start_time > time:
            self.turn_locked_on = True
            rospy.loginfo("locked on")

    def perform_left_turn_apriltag(self, gray, tag, radius = None):
        self.enable_lane_following = False
        self.enable_red_detection = False
        if self.left_turn_start_time is None: 
            self.left_turn_start_time = rospy.Time.now().to_sec()
            self.init_ticks_left = self._ticks_left
            self.init_ticks_right = self._ticks_right
        left_dist = self._ticks_left - self.init_ticks_left
        right_dist = self._ticks_right - self.init_ticks_right

        if radius is not None:
            dist_ratio = (radius - 0.05) / (radius + 0.05)
            
            diffr = abs(left_dist) - abs(right_dist)*dist_ratio

            corr = self.tick_pid.compute(diffr)

            # rospy.loginfo(f"diff: {diff} corr: {corr}")

            self.publish_cmd((0.5 * (radius - 0.05) / radius) - corr, (0.5 * (radius + 0.05) / radius) + corr)

            detections  = self.detector.detect(gray)
            center_diff = None

            for det in detections:
                if det.tag_id == tag:
                    tag_center_x = det.center[0]            
                    center_diff = tag_center_x - 320
                    break

        # rospy.loginfo(lane_error)
        if center_diff and abs(center_diff) < 50:
            self.turn_locked_on = True
            rospy.loginfo("locked on")

    def dynamic_control(self, distance_left, distance_right, max_time):
        self.enable_lane_following = False
        self.enable_red_detection = False
        if self.dynamic_control_start_time is None:
            self.dynamic_control_start_time = rospy.Time.now().to_sec()
            self.init_ticks_left = self._ticks_left
            self.init_ticks_right = self._ticks_right
            rospy.loginfo("starting dynamic control")
            rospy.loginfo(f"self.init_ticks_left: {self.init_ticks_left}, self.init_ticks_right: {self.init_ticks_right}")
            rospy.loginfo(f"self._ticks_left: {self._ticks_left}, self._ticks_right: {self._ticks_right}")


        dist_ratio = distance_left/distance_right

        left_dist = self._ticks_left - self.init_ticks_left
        right_dist = self._ticks_right - self.init_ticks_right
        
        diff = abs(left_dist) - abs(right_dist)*dist_ratio

        corr = self.dynamic_pid.compute(diff)

        # rospy.loginfo(f"diff: {diff} corr: {corr}")

        self.publish_cmd(0.4*dist_ratio * (1 - corr), 0.4 / dist_ratio * (1 + corr))

        if (rospy.Time.now().to_sec() - self.dynamic_control_start_time > max_time) or \
        ((abs(compute_distance(self._ticks_left - self.init_ticks_left)) > abs(distance_left)) and \
        (abs(compute_distance(self._ticks_right - self.init_ticks_right)) > abs(distance_right))):
            rospy.loginfo("ending dynamic control")
            rospy.loginfo(f"self.init_ticks_left: {self.init_ticks_left}, self.init_ticks_right: {self.init_ticks_right}")
            rospy.loginfo(f"self._ticks_left: {self._ticks_left}, self._ticks_right: {self._ticks_right}")
            self.dynamic_control_finished = True
            self.dynamic_control_start_time = None
            self.movements += 1
            rospy.loginfo(f"movements: {self.movements}")
            rospy.sleep(0.25)

    def turn_left_45(self):
        """
        Commands the robot to turn approximately 45 degrees to the left.
        Adjust the velocities and duration according to your robot's kinematics.
        """
        rospy.loginfo("Turning 45 degrees to the left.")
        turn_duration = 0.5  # Duration in seconds (adjust as needed)
        start_time = rospy.Time.now()
        rate = rospy.Rate(50)
        while rospy.Time.now() - start_time < rospy.Duration(1):
            stop_cmd = WheelsCmdStamped()
            stop_cmd.header.stamp = rospy.Time.now()
            stop_cmd.vel_left = 0
            stop_cmd.vel_right = 0
            self._publisher.publish(stop_cmd)
            rate.sleep()
        while rospy.Time.now() - start_time < rospy.Duration(turn_duration + 1):
            cmd = WheelsCmdStamped()
            cmd.header.stamp = rospy.Time.now()
            # For a left turn: slower left wheel and faster right wheel.
            cmd.vel_left = -0.6
            cmd.vel_right = 0
            self._publisher.publish(cmd)
            rate.sleep()
        while rospy.Time.now() - start_time < rospy.Duration(turn_duration + 2.75):
            cmd = WheelsCmdStamped()
            cmd.header.stamp = rospy.Time.now()
            # For a left turn: slower left wheel and faster right wheel.
            cmd.vel_left = 0.4
            cmd.vel_right = 0.4
            self._publisher.publish(cmd)
            rate.sleep()
        # Stop after turn.
        stop_cmd = WheelsCmdStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.vel_left = 0
        stop_cmd.vel_right = 0
        self._publisher.publish(stop_cmd)


    def perform_park(self, gray, tag):
        # Disable other behaviors while parking
        self.enable_lane_following = False
        self.enable_red_detection = False

        # Initialize on first entry
        if self.park_start_time is None:
            self.park_start_time = rospy.Time.now().to_sec()

        # Elapsed time since parking started
        elapsed = rospy.Time.now().to_sec() - self.park_start_time

        # AprilTag detection for centering and bottom-bound distance
        detections  = self.detector.detect(gray)
        tag_detected = False
        center_diff = None
        bottom_diff = None

        for det in detections:
            if det.tag_id == tag:
                tag_center_x = det.center[0]
                tag_center_y = det.center[1]                
                center_diff = tag_center_x - 320
                bottom_diff = 480 - tag_center_y
                
                tag_detected = True
                break

        if tag_detected: 
            corr = self.tagpark_pid.compute(center_diff)
            self.publish_cmd(0.3 - corr, 0.3 + corr)
        else:
            self.publish_cmd(-0.3, 0.3)
        

        # If in straight drive and tag near bottom (<200px), finish parking
        if tag_detected and bottom_diff is not None and bottom_diff < 250:
            rospy.loginfo(f"Tag {tag} close to bottom (Δy={bottom_diff:.1f}px). Parking complete.")
            self.movements += 1
            self.stop_robot()
            # invoke system shutdown
            #os.system("sudo shutdown -h now")

    def perform_park_lane_following(self, image, gray, tag, side, dt):
        self.enable_lane_following = False
        self.enable_red_detection = False
        # Initialize on first entry
        if self.park_start_time is None:
            self.park_start_time = rospy.Time.now().to_sec()

        # Elapsed time since parking started
        elapsed = rospy.Time.now().to_sec() - self.park_start_time

        # AprilTag detection for centering and bottom-bound distance
        detections  = self.detector.detect(gray)
        error = self.detect_lane_fast_wide(image, side)
        lane_corr = self.lane_pid_centroid.compute(error, dt)
        tag_detected = False
        center_diff = None
        bottom_diff = None

        for det in detections:
            if det.tag_id == tag:
                tag_center_x = det.center[0]
                tag_center_y = det.center[1]                
                center_diff = tag_center_x - 320
                bottom_diff = 480 - tag_center_y
                
                tag_detected = True
                break
            
        
        self.publish_cmd(0.4 - lane_corr, 0.4 + lane_corr)
        

        # If in straight drive and tag near bottom (<200px), finish parking
        if tag_detected and bottom_diff is not None and bottom_diff < 260:
            rospy.loginfo(f"Tag {tag} close to bottom (Δy={bottom_diff:.1f}px). Parking complete.")
            self.movements += 1
            self.stop_robot()
            # invoke system shutdown
            #os.system("sudo shutdown -h now")

    def set_led_color(self, r, g, b):
        num_leds = 5  # Most Duckiebots have 5 LEDs. Adjust if yours differs!

        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()

        color_msg = ColorRGBA()
        color_msg.r = r
        color_msg.g = g
        color_msg.b = b
        color_msg.a = 1
        pattern_msg.rgb_vals = [color_msg] * num_leds
        # Publish
        self._led_pub.publish(pattern_msg)

    def callback(self, msg, left_encoder_msg, right_encoder_msg):
        # 1) Update encoder readings and timing
        self._ticks_left = left_encoder_msg.data
        self._ticks_right = right_encoder_msg.data

        current = rospy.Time.now()
        dt = (current - self.last_time).to_sec()
        self.last_time = current

        # 2) Convert and undistort image, then preprocess
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed, gray = self.preprocess_image(undistorted)

        # if self.red_line_count == 0:
        #     self.set_led_color(1,1,1)
        
        # 4) Duck–blue detection (guarded)
        if self.enable_duckblue_detection:
            db_detected, self.dist_error, left_dist, right_dist = self.detect_duckblue(undistorted)
        else:
            db_detected, self.dist_error, left_dist, right_dist = None, None, None, None

        if self.red_line_count == 3 and self.enable_apriltag_detection:
            detections = self.detector.detect(gray)
            if detections:
                self.enable_apriltag_detection = False
                self.tagid = detections[0].tag_id

        # if db_detected and self.red_line_count < 4:
        #     if self.led_color != "green":
        #         self.led_color = "green"
        #         self.set_led_color(0, 1, 0)
        # else:
        #     if self.led_color != "blue":
        #         self.led_color = "blue"
        #         self.set_led_color(0, 0, 1)

        # 5) Red-line detection & handling
        if self.enable_red_detection and (current - self.last_red_stop_time) > self.red_cooldown:
            red_detected, red_dist = self.detect_red_line(preprocessed)
            if red_detected and red_dist is not None and red_dist < 0.15:
                # stop & record
                self.last_red_stop_time = current
                self.red_line_count += 1
                rospy.loginfo(f"red line count: {self.red_line_count}")
                self.stop_robot()
                if self.red_line_count == 1: rospy.sleep(1)
                rospy.sleep(0.5)
                if self.red_line_count < 5:
                    dot_detection_active = False
                # per-line decisions
                if self.red_line_count == 1:
                    _, _, left_dist, right_dist = self.detect_duckblue(undistorted)
                    rospy.loginfo(f"left_dist: {left_dist}, right_dist: {right_dist}")
                    if left_dist is not None and right_dist is not None:
                        if left_dist > right_dist + 10:
                            self.first_line_do_turn = False
                        else:
                            rospy.loginfo("Turning left")
                            self.first_line_do_turn = True
                elif self.red_line_count == 2:
                    self.left_turn_start_time = None
                    self.turn_locked_on = False

                elif self.red_line_count == 3:
                    self.left_turn_start_time = None
                    self.turn_locked_on = False
                    

                elif self.red_line_count == 4:
                    self.left_turn_start_time = None
                    self.turn_locked_on = False
                    # april-tag logic
                    if self.tagid and self.tagid == 48:
                        self.fourth_line_do_turn = True
                    else:
                        self.fourth_line_do_turn = False

                elif self.red_line_count == 5:
                    self.enable_blue_detection = True
                    self.enable_duckblue_detection = False
                    self.left_turn_start_time = None
                    self.turn_locked_on = False
                    # disable duck–blue and enter blue-phase
                    
                    # self.enable_blue_phase = True
                    rospy.loginfo("Entering blue-line phase")
                elif self.red_line_count == 6:
                    self.left_turn_start_time = None
                    self.turn_locked_on = False

        # 6) Execute any pending red-line turn
        if self.red_line_count == 1 and self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 0.55)
        elif self.red_line_count == 2 and self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, None, 2)
        elif self.red_line_count == 3 and not self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 0.55)
        elif self.red_line_count == 4 and self.fourth_line_do_turn and not self.turn_locked_on:
            self.enable_duckblue_detection = False
            self.perform_left_turn(preprocessed, 0.55)
        elif self.red_line_count == 5 and self.fourth_line_do_turn and not self.turn_locked_on:
            self.enable_blue_phase = True
        elif self.red_line_count == 5 and not self.fourth_line_do_turn and not self.turn_locked_on:
            self.enable_blue_phase = True
            self.perform_left_turn(preprocessed, 0.55)
            
        # elif self.red_line_count == 10 and self.park == 1:
        #     self.enable_blue_phase = False
        #     if self.turn_locked_on is False:
        #         self.perform_left_turn_apriltag(gray, 44, -0.45)
        #     else:
        #         if self.movements == 1:
        #             return
        #         self.perform_park(gray, 44)

        elif self.red_line_count == 6 and self.park == 1:
            self.enable_blue_phase = False
            if self.movements == 0:
                self.perform_park_lane_following(preprocessed, gray, 44, "right", dt)
            elif self.movements == 1:
                self.stop_robot()
                rospy.signal_shutdown("125/125")

        # elif self.red_line_count == 10 and self.park == 2:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.dynamic_control(0.2, 0.2, 5)
        #     if self.movements == 1:
        #         if self.turn_locked_on is False:
        #             self.perform_left_turn_apriltag(gray, 58, -0.45)
        #         else:
        #             if self.movements == 2:
        #                 return
        #             self.perform_park(gray, 58)

        elif self.red_line_count == 6 and self.park == 2:
            self.enable_blue_phase = False
            if self.movements == 0:
                self.dynamic_control(0.35, 0.35, 5)
            elif self.turn_locked_on is False and self.movements == 1:
                self.perform_left_turn_park(preprocessed, "left", -0.35, 0.5)
            else:
                if self.movements == 1:
                    self.perform_park_lane_following(preprocessed, gray, 58, "left", dt)
                elif self.movements == 2:
                    rospy.signal_shutdown("125/125")

        # elif self.red_line_count == 10 and self.park == 3:
        #     self.enable_blue_phase = False
        #     if self.turn_locked_on is False:
        #         self.perform_left_turn_apriltag(gray, 13, 0.45)
        #     else:
        #         if self.movements == 1:
        #             return
        #         self.perform_park(gray, 13)

        elif self.red_line_count == 6 and self.park == 3:
            self.enable_blue_phase = False
            if self.turn_locked_on is False:
                self.perform_left_turn_park(preprocessed, "left", 0.3, 2)
            else:
                if self.movements == 0:
                    self.perform_park_lane_following(preprocessed, gray, 13, "left", dt)
                elif self.movements == 1:
                    rospy.signal_shutdown("125/125")

        # elif self.red_line_count == 10 and self.park == 4:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.dynamic_control(0.2, 0.2, 5)
        #     if self.movements == 1:
        #         if self.turn_locked_on is False:
        #             self.perform_left_turn_apriltag(gray, 47, 0.45)
        #         else:
        #             if self.movements == 2:
        #                 return
        #             self.perform_park(gray, 47)

        elif self.red_line_count == 6 and self.park == 4:
            self.enable_blue_phase = False
            if self.turn_locked_on is False:
                self.perform_left_turn_park(preprocessed, "right", 0.6, 2)
            else:
                if self.movements == 0:
                    self.perform_park_lane_following(preprocessed, gray, 47, "right", dt)
                elif self.movements == 1:
                    rospy.signal_shutdown("125/125")
        

        # elif self.red_line_count == 6 and self.park == 1:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.align_with_red_line(preprocessed, dt)
        #     elif self.movements == 1:
        #         self.dynamic_control(0.025, 0.025, 5)
        #     elif self.movements == 2:
        #         self.dynamic_control(0.48, 0.36, 5)
        #     elif self.movements == 3:
        #         self.perform_park(gray, 44, 5)
        #     elif self.movements == 4:
        #         self.stop_robot()
        
        # elif self.red_line_count == 6 and self.park == 2:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.dynamic_control(0.25, 0.25, 5)
        #     elif self.movements == 1:
        #         self.dynamic_control(0.48, 0.36, 5)
        #     elif self.movements == 2:
        #         self.stop_robot()

        # elif self.red_line_count == 6 and self.park == 3:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.dynamic_control(0.36, 0.48, 5)
        #     elif self.movements == 1:
        #         self.dynamic_control(0.4, 0.4, 5)
        #     elif self.movements == 2:
        #         self.stop_robot()

        # elif self.red_line_count == 6 and self.park == 4:
        #     self.enable_blue_phase = False
        #     if self.movements == 0:
        #         self.dynamic_control(0.25, 0.25, 5)
        #     elif self.movements == 1:
        #         self.dynamic_control(0.36, 0.48, 5)
        #     elif self.movements == 2:
        #         self.dynamic_control(0.4, 0.4, 5)
        #     elif self.movements == 3:
        #         self.stop_robot()



        elif self.red_line_count == 100 and self.aligning_stopped is False:
            self.enable_lane_following = False
            self.enable_red_detection = False
            self.align_with_red_line(preprocessed)


        # 7) Blue-line phase detection & handling
        if self.enable_blue_phase:
            # Handle duckiebot yellow lane timeout
            if self.use_yellow_lane and self.lane_switch_start_time is not None:
                yellow_lane_elapsed = (current - self.lane_switch_start_time).to_sec()
                if yellow_lane_elapsed > 2.5:  # 3 second timeout for yellow lane
                    rospy.loginfo("4s elapsed. Switching back to white lane.")
                    self.use_yellow_lane = False
                    self.enable_blue_detection = True
                    self.lane_switch_start_time = None
            
            # Check for blue line detection (only when enabled)
            if self.enable_blue_detection:
                blue_detected, blue_dist = self.detect_blue_line(preprocessed)
                if blue_detected and blue_dist is not None and blue_dist < 0.15:
                    self.blue_line_count += 1
                    rospy.loginfo(f"blue line #{self.blue_line_count}")
                    
                    # On first blue line: stop, wait, then start duck-wait
                    if self.blue_line_count in (1, 2):
                        self.stop_robot()
                        rospy.sleep(1)
                        self.waiting_for_ducks = True
                        self.enable_blue_detection = False  # Disable blue detection to ignore the second blue line
                        self.blue_line_ref_ticks = self._ticks_left  # Store reference encoder value
                        # Initially disable duckiebot detection until we pass the duck
                        self.enable_duckblue_detection = False
                        rospy.loginfo(f"First blue line detected at ticks {self.blue_line_ref_ticks}, waiting for ducks")
            
            # Handle duck detection and waiting
            if self.waiting_for_ducks:
                duck_present, _ = self.detect_duck(undistorted)
                if duck_present:
                    self.stop_robot()
                    rospy.loginfo("Duck detected, waiting...")
                    return
                else:
                    rospy.loginfo("Duck gone, resuming lane-following")
                    self.waiting_for_ducks = False
                    # Continue driving but keep blue detection off until we've moved enough
            
            # Check if we've moved enough ticks since detecting blue line
            if not self.enable_blue_detection and self.blue_line_ref_ticks is not None:
                ticks_moved = abs(self._ticks_left - self.blue_line_ref_ticks)
                
                # Re-enable duckiebot detection if we've moved enough ticks (250)
                if ticks_moved > 250 and not self.enable_duckblue_detection:
                    rospy.loginfo(f"Moved {ticks_moved} ticks since blue line, enabling duckiebot detection")
                    self.enable_duckblue_detection = True
                    self.blue_line_ref_ticks = None
            
            # Check for duckiebot in front using duckblue detection
            if self.enable_duckblue_detection:
                db_detected, db_dist_error, left_dist, right_dist = self.detect_duckblue(undistorted)
                # rospy.loginfo(f"db_detected: {db_detected}, db_dist_error: {db_dist_error}")
                
                # Handle duckiebot detection/avoidance
                if db_detected and db_dist_error is not None and db_dist_error < 5:  # 15cm threshold for avoidance
                    rospy.loginfo("Duckiebot detected at < 15cm, initiating avoidance")
                    
                    # Switch to yellow lane for avoidance
                    if not self.use_yellow_lane:
                        self.enable_duckblue_detection = False
                        self.turn_left_45()
                        rospy.loginfo("Switching to yellow lane for avoidance")
                        self.use_yellow_lane = True
                        self.lane_switch_start_time = rospy.Time.now()
                    
                    # Continue with lane following using yellow lane detection
                    return
                

        
        # 8) Normal lane-following
        if self.enable_lane_following:
            lane_error = self.detect_lane_fast(preprocessed)
            dist_corr = self.dist_pid.compute(self.dist_error, dt) if self.dist_error is not None else None
            
            if lane_error is not None:
                lane_corr = self.lane_pid_centroid.compute(lane_error, dt)
                ls, rs = self.combine_error(lane_corr, dist_corr)
                self.publish_cmd(ls, rs)

        self.detect_lane_fast_wide(preprocessed, "left")




if __name__ == '__main__':
    node = LaneControllerNode('lane_controller_node')
    rospy.sleep(0.1)
    rospy.spin()
