import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

class SINDyPipeline:
    def __init__(self, 
                 poly_degree: int = 2,
                 alpha: float = 0.1,
                 threshold: float = 0.1,
                 window_length: int = 5,
                 polyorder: int = 2):
        """
        Initialize the SINDy pipeline.
        
        Args:
            poly_degree: Degree of polynomial features
            alpha: Regularization parameter for STLSQ
            threshold: Threshold for STLSQ
            window_length: Window length for Savitzky-Golay filter
            polyorder: Polynomial order for Savitzky-Golay filter
        """
        self.poly_degree = poly_degree
        self.alpha = alpha
        self.threshold = threshold
        self.window_length = window_length
        self.polyorder = polyorder
        
        # Initialize SINDy model
        self.feature_library = PolynomialLibrary(degree=poly_degree)
        self.optimizer = STLSQ(alpha=alpha, threshold=threshold)
        self.model = None
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with processed data
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Handle missing data
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('Timestamp')
        
        return df
    
    def extract_state_and_control(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract state variables and control inputs from DataFrame.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Tuple of (state, control) arrays
        """
        # Extract state variables
        x = df[['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']].values
        
        # Extract control inputs
        u = df[['Vx', 'Vy', 'Vz', 'Yaw']].values
        
        return x, u
    
    def compute_derivatives(self, x: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Compute time derivatives using Savitzky-Golay filter.
        
        Args:
            x: State array
            dt: Time step
            
        Returns:
            Array of derivatives
        """
        # Apply Savitzky-Golay filter
        x_smooth = savgol_filter(x, 
                                window_length=self.window_length,
                                polyorder=self.polyorder,
                                axis=0)
        
        # Compute derivatives
        x_dot = np.gradient(x_smooth, dt, axis=0)
        
        return x_dot
    
    def fit_model(self, x: np.ndarray, x_dot: np.ndarray, u: np.ndarray) -> None:
        """
        Fit SINDy model with control inputs.
        
        Args:
            x: State array
            x_dot: State derivatives
            u: Control inputs
        """
        # Create SINDy model
        self.model = SINDy(
            feature_library=self.feature_library,
            optimizer=self.optimizer,
            feature_names=['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Vx', 'Vy', 'Vz', 'Yaw']
        )
        
        # Combine state and control for fitting
        xu = np.hstack([x, u])
        
        # Fit model
        self.model.fit(xu, x_dot, t=0.1)
        
    def print_equations(self) -> None:
        """Print the learned equations."""
        if self.model is not None:
            self.model.print()
        else:
            print("Model not fitted yet.")
    
    def visualize_results(self, x: np.ndarray, x_dot: np.ndarray, u: np.ndarray) -> None:
        """
        Visualize the results of the model fitting.
        
        Args:
            x: State array
            x_dot: State derivatives
            u: Control inputs
        """
        if self.model is None:
            print("Model not fitted yet.")
            return
        
        # Combine state and control
        xu = np.hstack([x, u])
        
        # Predict derivatives
        x_dot_pred = self.model.predict(xu)
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each state variable
        state_names = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        for i, (ax, name) in enumerate(zip(axes, state_names)):
            ax.plot(x_dot[:, i], label='True')
            ax.plot(x_dot_pred[:, i], '--', label='Predicted')
            ax.set_title(f'Derivative of {name}')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the fitted model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            print("Model not fitted yet.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model coefficients
        np.save(os.path.join(output_dir, 'model_coefficients.npy'), 
                self.model.coefficients())
        
        # Save feature names
        np.save(os.path.join(output_dir, 'feature_names.npy'),
                self.model.feature_names)
    
    def run_pipeline(self, csv_path: str, output_dir: str = None) -> None:
        """
        Run the complete SINDy pipeline.
        
        Args:
            csv_path: Path to the CSV file
            output_dir: Directory to save results (optional)
        """
        # Load and preprocess data
        df = self.load_data(csv_path)
        
        # Extract state and control
        x, u = self.extract_state_and_control(df)
        
        # Compute derivatives
        x_dot = self.compute_derivatives(x)
        
        # Fit model
        self.fit_model(x, x_dot, u)
        
        # Print equations
        self.print_equations()
        
        # Visualize results
        self.visualize_results(x, x_dot, u)
        
        # Save model if output directory provided
        if output_dir is not None:
            self.save_model(output_dir)

if __name__ == "__main__":
    # Example usage
    pipeline = SINDyPipeline()
    pipeline.run_pipeline(
        csv_path="output_data/pose_estimation_*.csv",
        output_dir="output_data/sindy_model"
    ) 