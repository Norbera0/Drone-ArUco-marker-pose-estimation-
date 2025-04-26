import cv2 as cv
import numpy as np
import os
from datetime import datetime

class CameraCalibrator:
    def __init__(self, chessboard_dim=(9, 6), square_size=24):
        self.chessboard_dim = chessboard_dim
        self.square_size = square_size
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.obj_3D = np.zeros((chessboard_dim[0] * chessboard_dim[1], 3), np.float32)
        self.obj_3D[:, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1, 2)
        self.obj_3D *= square_size
        
        self.obj_points_3D = []  # 3D points in real world space
        self.img_points_2D = []  # 2D points in image plane
        
    def setup_directories(self, camera_name):
        """Setup directories for calibration data and images"""
        script_directory = os.path.dirname(os.path.abspath(__file__))
        calib_data_path = os.path.join(script_directory, "..", "..", "calibration_data", camera_name)
        calib_images_path = os.path.join(calib_data_path, "calib_images")
        
        os.makedirs(calib_images_path, exist_ok=True)
        return calib_data_path, calib_images_path
    
    def capture_calibration_images(self, camera, num_images=20, delay=1):
        """Capture calibration images from camera"""
        calib_data_path, calib_images_path = self.setup_directories(camera.name)
        
        print(f"Capturing {num_images} calibration images...")
        for i in range(num_images):
            ret, frame = camera.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(calib_images_path, f"calib_{timestamp}.png")
                cv.imwrite(image_path, frame)
                print(f"Captured image {i+1}/{num_images}")
                time.sleep(delay)
    
    def calibrate_camera(self, camera_name):
        """Perform camera calibration using captured images"""
        calib_data_path, calib_images_path = self.setup_directories(camera_name)
        
        # Process all calibration images
        for filename in os.listdir(calib_images_path):
            if filename.endswith(".png"):
                image_path = os.path.join(calib_images_path, filename)
                image = cv.imread(image_path)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                
                ret, corners = cv.findChessboardCorners(gray, self.chessboard_dim, None)
                if ret:
                    self.obj_points_3D.append(self.obj_3D)
                    corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
                    self.img_points_2D.append(corners2)
        
        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            self.obj_points_3D, self.img_points_2D, gray.shape[::-1], None, None
        )
        
        # Save calibration data
        np.savez(
            os.path.join(calib_data_path, f"calib_{camera_name}.npz"),
            camMatrix=mtx,
            distCoef=dist,
            rVector=rvecs,
            tVector=tvecs
        )
        
        return mtx, dist, rvecs, tvecs
    
    def load_calibration(self, camera_name):
        """Load saved calibration data"""
        calib_data_path, _ = self.setup_directories(camera_name)
        data = np.load(os.path.join(calib_data_path, f"calib_{camera_name}.npz"))
        
        return {
            "camMatrix": data["camMatrix"],
            "distCoef": data["distCoef"],
            "rVector": data["rVector"],
            "tVector": data["tVector"]
        } 