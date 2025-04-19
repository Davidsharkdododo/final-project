#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown.dtros import DTROS, NodeType


class WhiteContourDebugNode(DTROS):
    def __init__(self, node_name):
        super(WhiteContourDebugNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        
        # Get vehicle name and camera topic
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
        # Camera calibration parameters from your reference
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        

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


        self.white_range = self.hsv_ranges["red2"]
        
        # CvBridge and publisher/subscriber setup
        self.bridge = CvBridge()
        
        # Create publishers for debugging images
        self.contour_topic = f"/{self._vehicle_name}/white_contour_debug/contours"
        self.mask_topic = f"/{self._vehicle_name}/white_contour_debug/mask"
        self.raw_topic = f"/{self._vehicle_name}/white_contour_debug/raw"
        
        self.pub_contour = rospy.Publisher(self.contour_topic, Image, queue_size=1)
        self.pub_mask = rospy.Publisher(self.mask_topic, Image, queue_size=1)
        self.pub_raw = rospy.Publisher(self.raw_topic, Image, queue_size=1)
        
        # Subscribe to camera topic
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
        
        rospy.loginfo(f"[{self.node_name}] Initialized. Debugging white contours from camera images.")

    def undistort_image(self, image):
        """Undistort image using calibration parameters."""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        """Preprocess image by resizing and blurring."""
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def detect_white_contours(self, image):
        """Detect white contours in the given image."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask using white color range
        lower, upper = self.white_range
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area if needed
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        return mask, filtered_contours

    def create_debug_visualization(self, image, contours):
        """Create visualization with contours drawn on the image."""
        debug_img = image.copy()
        
        # Draw all contours
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        
        # Draw center line for reference
        h, w = image.shape[:2]
        cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        
        # Add contour count
        cv2.putText(debug_img, f"Contours: {len(contours)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_img

    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert compressed image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            # Undistort and preprocess
            undistorted = self.undistort_image(cv_image)
            preprocessed = self.preprocess_image(undistorted)
            
            # Detect white contours
            mask, contours = self.detect_white_contours(preprocessed)
            
            # Create debug visualization
            debug_img = self.create_debug_visualization(preprocessed, contours)
            
            # Publish debug images
            contour_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            raw_msg = self.bridge.cv2_to_imgmsg(preprocessed, encoding="bgr8")
            
            contour_msg.header.stamp = rospy.Time.now()
            mask_msg.header.stamp = rospy.Time.now()
            raw_msg.header.stamp = rospy.Time.now()
            
            self.pub_contour.publish(contour_msg)
            self.pub_mask.publish(mask_msg)
            self.pub_raw.publish(raw_msg)
            
            # Optional: Log info
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                rospy.logdebug(f"Found {len(contours)} white contours. Largest area: {largest_area:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def on_shutdown(self):
        """Clean shutdown procedure."""
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == '__main__':
    node = WhiteContourDebugNode('contours')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("Shutting down white contour debug node")