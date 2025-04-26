import cv2 as cv
import numpy as np
from djitellopy import Tello
import threading
import time
from datetime import datetime
import os
import csv

from ..utils.keyboard_handler import KeyboardHandler
from ..pose_estimation.pose_estimator import PoseEstimator
from ..calibration.camera_calibration import CameraCalibrator

class DroneController:
    def __init__(self, camera_mode="tello"):
        self.camera_mode = camera_mode
        self.drone = None
        self.keyboard = KeyboardHandler()
        self.pose_estimator = PoseEstimator()
        self.camera_calibrator = CameraCalibrator()
        
        self.flight_active = False
        self.new_frame_available = False
        self.frame = None
        
        # Load camera calibration data
        self.calibration_data = self.camera_calibrator.load_calibration(camera_mode)
        
    def setup_directories(self):
        """Setup required directories"""
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_directory, "..", "..", "output_data")
        self.input_dir = os.path.join(script_directory, "..", "..", "input_data")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
    
    def initialize_drone(self):
        """Initialize drone connection"""
        if self.camera_mode == "tello":
            self.drone = Tello()
            self.drone.connect()
            print(f"Battery: {self.drone.get_battery()}%")
            self.drone.streamon()
    
    def pose_estimation_thread(self):
        """Thread for continuous pose estimation"""
        while True:
            if self.drone is not None and self.new_frame_available:
                frame_read = self.drone.get_frame_read()
                b, g, r = cv.split(frame_read.frame)  # Fix blue-red interchange
                frame = cv.merge((r, g, b))
                
                # Estimate pose
                frame, poses = self.pose_estimator.estimate_pose(
                    frame,
                    self.calibration_data["camMatrix"],
                    self.calibration_data["distCoef"],
                    self.drone
                )
                
                # Log pose data
                if poses:
                    self.pose_estimator.log_pose_data(poses)
                
                self.frame = frame
                self.new_frame_available = False
    
    def control_thread(self):
        """Thread for drone control"""
        while True:
            if self.flight_active and self.drone is not None:
                # Get keyboard inputs
                control_inputs = self.keyboard.get_control_inputs()
                
                # Send control commands to drone
                self.drone.send_rc_control(*control_inputs)
                
                # Check for landing command
                if self.keyboard.should_land():
                    self.drone.land()
                    self.flight_active = False
                
                time.sleep(0.1)
    
    def run(self):
        """Main execution loop"""
        self.setup_directories()
        self.initialize_drone()
        self.keyboard.init()
        
        # Start threads
        pose_thread = threading.Thread(target=self.pose_estimation_thread)
        control_thread = threading.Thread(target=self.control_thread)
        
        pose_thread.daemon = True
        control_thread.daemon = True
        
        pose_thread.start()
        control_thread.start()
        
        # Main loop
        while True:
            if self.frame is not None:
                cv.imshow("Drone View", self.frame)
            
            # Check for takeoff command
            if self.keyboard.should_takeoff() and not self.flight_active:
                self.drone.takeoff()
                time.sleep(2)
                self.flight_active = True
            
            # Exit on 'q' press
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.drone is not None:
            self.drone.land()
            self.drone.streamoff()
        cv.destroyAllWindows()

if __name__ == "__main__":
    controller = DroneController()
    controller.run() 