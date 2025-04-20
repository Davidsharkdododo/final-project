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

        self.dist_error = None
        self.first_line_do_turn = True  # track first red line decision need to chnage back to None
        self.fourth_line_do_turn = False
        self.fourth_line_do_turn = False

        self.turn_locked_on = False
        self.left_turn_start_time = None

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
        self.blue_line_count      = 2
        self.last_blue_stop_time  = rospy.Time(0)
        self.blue_cooldown        = rospy.Duration(10)   # 3 s ignore between blue detections
        self.enable_blue_phase    = True
        self.park = 4   #could also be 1, 2,3,4

        self.last_duckiebot_scan_time = rospy.Time(0)
        
        self.detector = Detector(families="tag36h11")
        self.last_detection_time = rospy.Time(0)
        self.detection_interval = 1  # seconds

        self.base_speed = 0.5
        
        # PID controllers
        self.lane_pid = PID(-0.22, 0.001, 0.002, "P", -0.2, 0.3)
        self.dist_pid = PID(0.06, 0.001, 0.002, "P", -self.base_speed, 0.2)
        self.tick_pid = PID(0.02, 0.001, 0.002, "P", -0.3, 0.3)
        self.tag_pid = PID(0.005, 0.0, 0.001, "P", -0.3, 0.3)
        
        self.last_time = rospy.Time.now()
        self.target_april_tag = 47 # could also be 47, 13, 44, 58
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
        self.red_line_count = 6
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
            "blue": (np.array([100, 110, 100]), np.array([140, 255, 255])),
            "duck": (np.array([6, 82, 108]),   np.array([22, 255, 255])),
            "duckblue": (np.array([50, 90, 0]),   np.array([110, 255, 255]))
        }
        
        # CvBridge and publishers/subscribers
        self.bridge = CvBridge()
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        self.debug_gray_topic = f"/{self._vehicle_name}/camera_node/image/compressed/gray"
        self.pub_debug_gray = rospy.Publisher(self.debug_gray_topic, Image, queue_size=1)
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
        """
        Fast lane detection using fixed-height scanning.
        Scans for white pixels in the bottom half of the image.
        """
        debug = False

        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        lane_color = "yellow" if self.use_yellow_lane else "white"
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges[lane_color][0], self.hsv_ranges[lane_color][1])
        
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights.
        # scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        scan_heights = [int(height * pct) for pct in [0.70, 0.40, 0.20, 0.10, 0]]
        if debug:
            output = image.copy()
            bottom_half_out = output[half_height_start:full_height, :]
        
        # base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        base_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        offsets = []
        valid_detections = []
        
        for i, y in enumerate(scan_heights):
            for x in range(center_x, width, 50):
                if white_mask[y, x] > 0:
                    for x2 in range(x, x - 50, -5):
                        if not white_mask[y, x2] > 0:
                            offsets.append(x2 + 5 - center_x)
                            valid_detections.append(i)
                            if debug: cv2.circle(bottom_half_out, (x2, y), 3, (0, 255, 0), -1)                    
                            break
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
                error = error_raw * 0.01 - 1.5
            
            if debug:
                color = (0, 0, 255) if len(valid_detections) < 3 else (0, 255, 0)
                cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                debug_img_msg = self.bridge.cv2_to_imgmsg(bottom_half_out, encoding="bgr8")
                debug_img_msg.header.stamp = rospy.Time.now()
                self.pub_lane.publish(debug_img_msg)
        else:
            error = 10
            # rospy.loginfo("error is none")
        
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
        self._publisher.publish(cmd)

    def perform_left_turn(self, image, time, radius = None):
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
        if abs(lane_error) < 0.5 and rospy.Time.now().to_sec() - self.left_turn_start_time > time:
            self.enable_lane_following = True
            self.enable_red_detection = True
            self.turn_locked_on = True
            rospy.loginfo("locked on")

    def perform_park(self, image, time, tag, radius):
        # Disable other behaviors while parking
        self.enable_lane_following = False
        self.enable_red_detection = False

        # Initialize on first entry
        if not hasattr(self, 'park_start_time') or self.park_start_time is None:
            self.park_start_time = rospy.Time.now().to_sec()
            self.init_ticks_left = self._ticks_left
            self.init_ticks_right = self._ticks_right
            self.straight_drive_active = False

        # Elapsed time since parking started
        elapsed = rospy.Time.now().to_sec() - self.park_start_time

        # If time limit reached, force straight-driving phase
        if elapsed > time and not self.straight_drive_active:
            rospy.loginfo(f"Time {time}s elapsed; entering straight drive.")
            self.straight_drive_active = True

        # Compute initial turn speeds based on encoder ticks and fixed radius
        left_dist  = self._ticks_left  - self.init_ticks_left
        right_dist = self._ticks_right - self.init_ticks_right
        dist_ratio = (radius - 0.05) / (radius + 0.05)
        diff       = abs(left_dist) - abs(right_dist) * dist_ratio
        corr       = self.tick_pid.compute(diff)
        base_left  = (0.5 * (radius - 0.05) / radius) - corr
        base_right = (0.5 * (radius + 0.05) / radius) + corr

        # AprilTag detection for centering and bottom-bound distance
        gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections  = self.detector.detect(gray)
        tag_detected = False
        center_diff = None
        bottom_diff = None

        for det in detections:
            if det.tag_id == tag:
                # horizontal centering error
                xs = det.corners[:, 0]
                left_px = xs.min()
                right_px = xs.max()
                center_diff = abs(left_px - (image.shape[1] - right_px))
                # vertical bottom-bound distance
                ys = det.corners[:, 1]
                bottom_diff = image.shape[0] - ys.max()
                tag_detected = True
                break

        # If centered horizontally (<10px) and within time, switch to straight drive
        if not self.straight_drive_active and tag_detected and center_diff is not None and center_diff < 10:
            rospy.loginfo(f"Tag {tag} centered (Δpx={center_diff:.1f}). Entering straight drive.")
            self.straight_drive_active = True

        # Choose speeds: turn until straight drive, else straight
        if self.straight_drive_active:
            # Encoder-based straight heading correction
            tick_error = self._ticks_left - self._ticks_right
            corr_straight = self.tick_pid.compute(tick_error)
            left_speed  = 0.4 - corr_straight
            right_speed = 0.4 + corr_straight
        else:
            left_speed  = base_left
            right_speed = base_right

        self.publish_cmd(left_speed, right_speed)

        # If in straight drive and tag near bottom (<200px), finish parking
        if self.straight_drive_active and tag_detected and bottom_diff is not None and bottom_diff < 200:
            rospy.loginfo(f"Tag {tag} close to bottom (Δy={bottom_diff:.1f}px). Parking complete.")
            self.stop_robot()
            # invoke system shutdown
            #os.system("sudo shutdown -h now")


    def turn_left_45(self):
        """
        Commands the robot to turn approximately 45 degrees to the left.
        Adjust the velocities and duration according to your robot's kinematics.
        """
        rospy.loginfo("Turning 45 degrees to the left.")
        turn_duration = 0.5  # Duration in seconds (adjust as needed)
        start_time = rospy.Time.now()
        rate = rospy.Rate(50)
        while rospy.Time.now() - start_time < rospy.Duration(turn_duration):
            cmd = WheelsCmdStamped()
            cmd.header.stamp = rospy.Time.now()
            # For a left turn: slower left wheel and faster right wheel.
            cmd.vel_left = -0.6
            cmd.vel_right = 0
            self._publisher.publish(cmd)
            rate.sleep()
        # Stop after turn.
        stop_cmd = WheelsCmdStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.vel_left = 0
        stop_cmd.vel_right = 0
        self._publisher.publish(stop_cmd)

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

        detections = self.detector.detect(gray)
        # 4) Duck–blue detection (guarded)
        if self.enable_duckblue_detection:
            db_detected, self.dist_error, left_dist, right_dist = self.detect_duckblue(undistorted)
        else:db_detected, self.dist_error, left_dist, right_dist = None, None, None, None

        # 5) Red-line detection & handling
        if self.enable_red_detection and (current - self.last_red_stop_time) > self.red_cooldown:
            red_detected, red_dist = self.detect_red_line(preprocessed)
            if red_detected and red_dist is not None and red_dist < 0.15:
                # stop & record
                self.last_red_stop_time = current
                self.red_line_count += 1
                rospy.loginfo(f"red line count: {self.red_line_count}")
                self.stop_robot()
                rospy.sleep(0.5)
                if self.red_line_count < 5:
                    dot_detection_active = False
                # per-line decisions
                if self.red_line_count == 1:
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
                    if detections and detections[0].tag_id == 48:
                        self.fourth_line_do_turn = True
                    else:
                        self.fourth_line_do_turn = False

                elif self.red_line_count == 5:
                    self.left_turn_start_time = None
                    self.turn_locked_on = False
                    # disable duck–blue and enter blue-phase
                    
                    self.enable_blue_phase = True
                    self.enable_red_detection = False
                    rospy.loginfo("Entering blue-line phase")

        # 6) Execute any pending red-line turn
        if self.red_line_count == 1 and self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 0.55)
        elif self.red_line_count == 2 and self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 1)
        elif self.red_line_count == 3 and not self.first_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 3, 0.55)
        elif self.red_line_count == 4 and self.fourth_line_do_turn and not self.turn_locked_on:
            self.perform_left_turn(preprocessed, 3, 0.55)
        elif self.red_line_count == 5:
            self.enable_duckblue_detection = False
            self.enable_red_detection = False
        # elif self.red_line_count == 5 and not self.fourth_line_do_turn and not self.turn_locked_on:
        #     self.perform_left_turn(preprocessed, 0.55)

        elif self.red_line_count == 7 and self.park == 1:
            self.perform_park(preprocessed, 1, 44, -0.25)
        elif self.red_line_count == 7 and self.park == 2:
            self.perform_park(preprocessed, 1, 58, -0.25)
        elif self.red_line_count == 7 and self.park == 3:
            self.perform_park(preprocessed, 2.5, 13, 0.55)   
        elif self.red_line_count == 7 and self.park == 4:
            self.perform_park(preprocessed, 2.5 , 47, 0.55)    

        # 7) Blue-line phase detection & handling
        if self.enable_blue_phase and (current - self.last_blue_stop_time) > self.blue_cooldown:
            blue_detected, blue_dist = self.detect_blue_line(preprocessed)
            if blue_detected and blue_dist is not None and blue_dist < 0.15:
                self.last_blue_stop_time = current
                self.blue_line_count += 1
                rospy.loginfo(f"blue line #{self.blue_line_count}")

                # on 1st and 3rd blue lines: stop, wait, then start duck-wait
                if self.blue_line_count in (1, 2):
                    self.stop_robot()
                    rospy.sleep(2)
                    self.waiting_for_ducks = True

        if self.waiting_for_ducks:
            duck_present, _ = self.detect_duck(undistorted)
            rospy.loginfo(duck_present)
            if duck_present:
                self.stop_robot()
                return
            else:
                self.waiting_for_ducks = False
                rospy.loginfo("Duck gone, resuming lane-following")            
        dot_detection_active = False 
        
        if self.blue_line_count >= 2: self.enable_red_detection = True

        if self.red_line_count >= 5: 
            if self.lane_switch_start_time is not None:
                elapsed_since_switch = current - self.lane_switch_start_time

                # after 3 s put lane colour back to white
                if elapsed_since_switch >= rospy.Duration(3.0) and self.use_yellow_lane:
                    rospy.loginfo("3s elapsed. Switching back to white lane.")
                    self.use_yellow_lane = False

                # keep dot detection OFF while we are in the cooldown window
                if elapsed_since_switch < self.lane_switch_cooldown:
                    dot_detection_active = False
                else:                                     # cooldown finished
                    self.lane_switch_start_time = None
                    dot_detection_active = True
            else:
                # no cooldown in progress → we may run dot detection
                dot_detection_active = True


         # --- Duckiebot detection logic ---
        if dot_detection_active:
            (detection, centers) = cv2.findCirclesGrid(
                undistorted,
                patternSize=tuple(self.circlepattern_dims),
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=self.simple_blob_detector,
            )
            if detection > 0 and centers is not None and centers.shape[0] >= 2:
                dot0 = centers[0, 0]
                dot1 = centers[1, 0]
                distance_dots = np.linalg.norm(dot0 - dot1)
                rospy.loginfo("Distance between first two dots: %.2f pixels", distance_dots)
                # Trigger lane switch if distance exceeds threshold.
                if distance_dots > 9:
                    rospy.loginfo("Distance > 13 detected. Stopping for 3 seconds, turning left 45°, and switching lane detection.")
                    stop_cmd = WheelsCmdStamped()
                    stop_cmd.header.stamp = rospy.Time.now()
                    stop_cmd.vel_left = 0
                    stop_cmd.vel_right = 0
                    self._publisher.publish(stop_cmd)
                    rospy.sleep(1)
                    # Turn left 45 degrees.
                    self.turn_left_45()
                    # Immediately switch lane detection to yellow.
                    self.use_yellow_lane = True
                    self.lane_switch_start_time = rospy.Time.now()
        # 8) Normal lane-following
        if self.enable_lane_following:
            lane_error = self.detect_lane_fast(preprocessed)
            dist_corr = self.dist_pid.compute(self.dist_error, dt) if self.dist_error is not None else None
            
            if lane_error is not None:
                lane_corr = self.lane_pid.compute(lane_error, dt)
                ls, rs = self.combine_error(lane_corr, dist_corr)
                self.publish_cmd(ls, rs)




if __name__ == '__main__':
    node = LaneControllerNode('lane_controller_node')
    rospy.sleep(0.1)
    rospy.spin()
