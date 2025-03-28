#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import Header, ColorRGBA
from duckietown.dtros import DTROS, NodeType

Wheel_rad = 0.0318

def compute_distance(ticks):
    rotations = ticks / 135
    return 2 * 3.1415 * Wheel_rad * rotations

def compute_ticks(distance):
    rotations = distance / (2 * 3.1415 * Wheel_rad)
    return rotations * 135

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"  
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"      
        # Queue to store detected lane colors
        self.q = []
        # Flags for maneuvers
        self.maneuver_active = False
        self.blue_stop_active = False
        self.duck_stop_active = False

        # Time lock for blue detection to ignore a second blue line.
        self.last_blue_stop_time = rospy.Time(0)
        self.blue_lock_duration = 4.0  # seconds

        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # HSV ranges for lane colors.
        self.hsv_ranges = {
            "blue": (np.array([100, 110, 100]), np.array([140, 255, 255])),
            "duck": (np.array([6, 82, 108]),   np.array([22, 255, 255]))
        }
        
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=10)
        self._ticks_left = 0
        self._ticks_right = 0

        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        self.publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        
        self.rate = rospy.Rate(3)
        
    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def compute_distance_homography(self, u, v):
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def detect_lane_color(self, image):
        """
        Process the image and return:
          - an annotated image,
          - the detected color (if any),
          - the computed distance (using the largest valid contour).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_color = None
        detected_distance = None
        output = image.copy()
        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    u = x + w / 2
                    v = y + h
                    distance = self.compute_distance_homography(u, v)
                    cv2.putText(output, f"{color}: {distance:.2f}m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    detected_color = color
                    detected_distance = distance
        return output, detected_color, detected_distance

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def dynamic_motor_control(self, left_power, right_power, distance_left, distance_right):
        """
        Drives the robot until the left and right wheels have traveled the respective distances.
        """
        rate = rospy.Rate(100)
        msg = WheelsCmdStamped()
        msg.vel_left = left_power
        msg.vel_right = right_power

        init_ticks_left = self._ticks_left
        init_ticks_right = self._ticks_right

        while (not rospy.is_shutdown()) and \
              (abs(compute_distance(self._ticks_left - init_ticks_left)) < abs(distance_left)) and \
              (abs(compute_distance(self._ticks_right - init_ticks_right)) < abs(distance_right)):

            # Optionally adjust speeds based on wheel distance error.
            dist_ratio = distance_left / distance_right if distance_right != 0 else 1
            left_dist = self._ticks_left - init_ticks_left
            right_dist = self._ticks_right - init_ticks_right
            diff = abs(left_dist) - abs(right_dist) * dist_ratio
            modifier = 0 * diff / 1000  # Tune as needed.

            msg.vel_left = left_power * (1 - modifier)
            msg.vel_right = right_power * (1 + modifier)
            msg.header.stamp = rospy.Time.now()
            self.publisher.publish(msg)
            rate.sleep()

        # Stop the robot.
        stop_msg = WheelsCmdStamped()
        stop_msg.header = Header()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.vel_left = 0
        stop_msg.vel_right = 0
        self.publisher.publish(stop_msg)

    def stop_robot(self):
        """Publish a stop command (zero velocity) to the wheels."""
        stop_msg = WheelsCmdStamped()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.vel_left = 0
        stop_msg.vel_right = 0
        self.publisher.publish(stop_msg)
        rospy.loginfo("Robot stopped.")

    def callback(self, msg):
        # Process each incoming image to update detection information.
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        color_image, detected_color, lane_distance = self.detect_lane_color(preprocessed)
        
        # Publish the undistorted and annotated images.
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="bgr8")
        self.pub_undistorted.publish(undistorted_msg)
        lane_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        self.pub_lane.publish(lane_msg)

        # --- Blue lane detection logic ---
        # When a blue line is detected within 20cm, queue it only if enough time has passed
        if detected_color == "blue":
            current_time = rospy.Time.now()
            time_since_last_blue = (current_time - self.last_blue_stop_time).to_sec()
            if lane_distance <= 0.20 and not self.blue_stop_active and time_since_last_blue > self.blue_lock_duration:
                rospy.loginfo("Detected blue line at %.3f m", lane_distance)
                self.q.append("blue")
                self.blue_stop_active = True
                self.maneuver_active = True

        # --- Duck detection logic ---
        if detected_color == "duck":
            if lane_distance < 0.20 and not self.duck_stop_active:
                rospy.loginfo("Detected duck at %.3f m", lane_distance)
                self.q.append("duck")
                self.duck_stop_active = True
                self.maneuver_active = True

    def run(self):
        """
        Main loop: if a lane color is queued, execute the corresponding maneuver.
        For a blue lane, the logic is:
          1. Stop for 2 seconds.
          2. Then, check if a duck is present within 20cm.
             - If a duck is detected, keep stopping (rechecking every 0.5 sec)
               until no duck is detected.
          3. Finally, resume forward motion.
          Also, once a blue stop is processed, a time lock is set to ignore subsequent blue detections.
        """
        while not rospy.is_shutdown():
            if self.q:
                # Peek at the first queued color.
                color = self.q[0]
                # Always stop the robot first.
                self.stop_robot()
                
                if color == "blue":
                    rospy.loginfo("Blue line detected: stopping for 2 seconds.")
                    rospy.sleep(1)
                    # Check for a duck within 20cm.
                    duck_present = False
                    try:
                        msg = rospy.wait_for_message(self._camera_topic, CompressedImage, timeout=1)
                        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                        undistorted = self.undistort_image(cv_img)
                        preprocessed = self.preprocess_image(undistorted)
                        _, curr_color, curr_distance = self.detect_lane_color(preprocessed)
                        if curr_color == "duck" and curr_distance < 0.20:
                            duck_present = True
                    except rospy.ROSException:
                        duck_present = False
                    
                    while duck_present and not rospy.is_shutdown():
                        rospy.loginfo("Duck detected within 20cm; continuing to stop...")
                        self.stop_robot()
                        rospy.sleep(0.5)
                        try:
                            msg = rospy.wait_for_message(self._camera_topic, CompressedImage, timeout=1)
                            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                            undistorted = self.undistort_image(cv_img)
                            preprocessed = self.preprocess_image(undistorted)
                            _, curr_color, curr_distance = self.detect_lane_color(preprocessed)
                            if curr_color != "duck" or curr_distance >= 0.20:
                                duck_present = False
                        except rospy.ROSException:
                            duck_present = False
                    
                    rospy.loginfo("No duck detected after blue line stop. Resuming forward motion.")
                    self.q.pop(0)
                    self.blue_stop_active = False
                    self.maneuver_active = False
                    # Update the time lock so that subsequent blue detections are ignored for a while.
                    self.last_blue_stop_time = rospy.Time.now()
                elif color == "duck":
                    rospy.loginfo("Duck maneuver: waiting until duck clears.")
                    duck_cleared = False
                    while not duck_cleared and not rospy.is_shutdown():
                        try:
                            msg = rospy.wait_for_message(self._camera_topic, CompressedImage, timeout=1)
                        except rospy.ROSException:
                            continue
                        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                        undistorted = self.undistort_image(cv_img)
                        preprocessed = self.preprocess_image(undistorted)
                        _, curr_color, curr_distance = self.detect_lane_color(preprocessed)
                        if curr_color != "duck" or curr_distance >= 0.20:
                            duck_cleared = True
                        else:
                            rospy.loginfo("Duck still detected within 20cm, waiting...")
                        rospy.sleep(0.5)
                    rospy.loginfo("Duck cleared, resuming movement.")
                    self.q.pop(0)
                    self.duck_stop_active = False
                    self.maneuver_active = False
            else:
                # No queued maneuvers: keep moving forward.
                self.dynamic_motor_control(0.45, 0.5, 0.05, 0.05)
            self.rate.sleep()

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
