# Drone Control and System Identification using SINDy algorithm

Implements a comprehensive system for drone control, pose estimation, and system identification using the Tello drone

## Project Structure

The project is organized into several main components:

### 1. Calibration (`src/calibration/`)
- `camera_calibration.py`: Handles camera calibration using a chessboard pattern
- Features:
  - Automatic chessboard detection
  - Intrinsic and extrinsic parameter estimation
  - Calibration data storage and loading

### 2. Control (`src/control/`)
- `drone_controller.py`: Main controller for the Tello drone
- Features:
  - Drone initialization and connection
  - Real-time control using keyboard inputs
  - Multi-threaded operation for control and pose estimation
  - Integration with pose estimation and camera calibration

### 3. Pose Estimation (`src/pose_estimation/`)
- `pose_estimator.py`: Handles ArUco marker detection and pose estimation
- Features:
  - ArUco marker detection and tracking
  - 6-DOF pose estimation
  - Real-time visualization of pose
  - Data logging for system identification

### 4. System Identification (`src/system_identification/`)
- `sindy_pipeline.py`: Implements Sparse Identification of Nonlinear Dynamics (SINDy)
- Features:
  - Data preprocessing and filtering
  - State and control input extraction
  - SINDy model fitting
  - Model visualization and evaluation
  - Model saving and loading

### 5. Utilities (`src/utils/`)
- `keyboard_handler.py`: Manages keyboard input for drone control
- Features:
  - Customizable control mapping
  - Real-time input handling
  - Takeoff and landing commands

## Directory Structure

```
.
├── src/
│   ├── calibration/
│   ├── control/
│   ├── pose_estimation/
│   ├── system_identification/
│   └── utils/
├── calibration_data/
├── input_data/
└── output_data/
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Norbera0/Drone-ArUco-marker-pose-estimation-.git
   cd Drone-ArUco-marker-pose-estimation-
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Hardware Requirements

- Tello drone
- Computer with WiFi capability
- Chessboard pattern for calibration (9x6 grid recommended)
- ArUco markers for pose estimation

## Usage

1. Camera Calibration:
   ```python
   from src.calibration.camera_calibration import CameraCalibrator
   calibrator = CameraCalibrator()
   calibrator.calibrate_camera("tello")
   ```

2. Drone Control:
   ```python
   from src.control.drone_controller import DroneController
   controller = DroneController()
   controller.run()
   ```

3. System Identification:
   ```python
   from src.system_identification.sindy_pipeline import SINDyPipeline
   pipeline = SINDyPipeline()
   pipeline.run_pipeline("output_data/pose_estimation_*.csv")
   ```

## Control Mapping

- `e`: Takeoff
- `q`: Land
- Arrow Keys: Control yaw and altitude
- `w/s`: Forward/backward
- `a/d`: Left/right

## Data Collection

The system automatically collects and stores:
- Camera calibration data
- Pose estimation data
- Control inputs
- System identification results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
