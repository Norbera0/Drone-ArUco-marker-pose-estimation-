import cv2 as cv
import numpy as np
import os
from datetime import datetime
import csv

class PoseEstimator:
    def __init__(self, marker_length=187, dictionary_type=cv.aruco.DICT_6X6_250):
        self.marker_length = marker_length
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary_type)
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
        # Define marker corners in 3D space
        self.obj_points = np.zeros((4, 1, 3), dtype=np.float32)
        self.obj_points[0][0] = [-marker_length/2.0, marker_length/2.0, 0]
        self.obj_points[1][0] = [marker_length/2.0, marker_length/2.0, 0]
        self.obj_points[2][0] = [marker_length/2.0, -marker_length/2.0, 0]
        self.obj_points[3][0] = [-marker_length/2.0, -marker_length/2.0, 0]
        
    def setup_output_directory(self):
        """Setup directory for pose estimation results"""
        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_directory, "..", "..", "output_data")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def estimate_pose(self, frame, camera_matrix, dist_coeffs, drone=None):
        """Estimate pose from ArUco markers in the frame"""
        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect markers
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        
        poses = []
        if marker_ids is not None:
            # Draw detected markers
            cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            
            # Estimate pose for each marker
            for i in range(len(marker_ids)):
                _, rvec, tvec = cv.solvePnP(
                    self.obj_points, 
                    marker_corners[i], 
                    camera_matrix, 
                    dist_coeffs, 
                    False, 
                    cv.SOLVEPNP_IPPE_SQUARE
                )
                
                # Get drone sensor data if available
                sensor_data = {}
                if drone is not None:
                    sensor_data = {
                        "pitch": drone.get_pitch(),
                        "roll": drone.get_roll(),
                        "yaw": drone.get_yaw(),
                        "tof": drone.get_distance_tof()
                    }
                
                # Calculate resultant vector
                resultant_vector = np.sqrt(tvec[0][0]**2 + tvec[1][0]**2 + tvec[2][0]**2)
                
                poses.append({
                    "marker_id": marker_ids[i][0],
                    "translation": tvec,
                    "rotation": rvec,
                    "resultant_vector": resultant_vector,
                    "sensor_data": sensor_data
                })
                
                # Draw coordinate axes
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        return frame, poses
    
    def log_pose_data(self, poses, control_inputs=None):
        """Log pose estimation data to CSV file"""
        output_dir = self.setup_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"pose_estimation_{timestamp}.csv")
        
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header if file is new
            if not file_exists:
                header = ["Timestamp", "Marker ID", "Tx", "Ty", "Tz", 
                         "Rx", "Ry", "Rz", "Resultant Vector"]
                if control_inputs is not None:
                    header.extend(["Vx", "Vy", "Vz", "Yaw"])
                writer.writerow(header)
            
            # Write pose data
            for pose in poses:
                row = [
                    datetime.now().strftime("%H:%M:%S.%f"),
                    pose["marker_id"],
                    pose["translation"][0][0],
                    pose["translation"][1][0],
                    pose["translation"][2][0],
                    pose["rotation"][0][0],
                    pose["rotation"][1][0],
                    pose["rotation"][2][0],
                    pose["resultant_vector"]
                ]
                
                if control_inputs is not None:
                    row.extend(control_inputs)
                
                writer.writerow(row) 