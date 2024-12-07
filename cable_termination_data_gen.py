import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import savemat
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import multiprocessing as mp
from tqdm import tqdm
from numba import jit
import os
import time

@jit(nopython=True)
def solve_poisson_fdm_numba(V, V_old, epsilon, dx2, dy2, n, max_iter, tol, voltage_bc):
    """
    Numba-optimized FDM solver for the 2D Poisson equation with voltage boundary conditions.
    """
    max_changes = np.zeros(max_iter)
    bc_mask = voltage_bc[0]
    bc_values = voltage_bc[1]
    
    for iter in range(max_iter):
        # Copy current solution to old solution
        for i in range(n):
            for j in range(n):
                V_old[i,j] = V[i,j]
        
        # Update interior points
        for i in range(1, n-1):
            for j in range(1, n-1):
                if not bc_mask[i,j]:  # Only update non-boundary points
                    eps_x_plus = (epsilon[i+1,j] + epsilon[i,j]) / 2
                    eps_x_minus = (epsilon[i-1,j] + epsilon[i,j]) / 2
                    eps_y_plus = (epsilon[i,j+1] + epsilon[i,j]) / 2
                    eps_y_minus = (epsilon[i,j-1] + epsilon[i,j]) / 2
                    
                    V[i,j] = (
                        (eps_x_plus * V[i+1,j] + eps_x_minus * V[i-1,j]) / dx2 +
                        (eps_y_plus * V[i,j+1] + eps_y_minus * V[i,j-1]) / dy2
                    ) / ((eps_x_plus + eps_x_minus) / dx2 + 
                         (eps_y_plus + eps_y_minus) / dy2)
        
        # Apply boundary conditions
        for i in range(n):
            for j in range(n):
                if bc_mask[i,j]:
                    V[i,j] = bc_values[i,j]
        
        # Calculate maximum change
        max_change = 0.0
        for i in range(n):
            for j in range(n):
                change = abs(V[i,j] - V_old[i,j])
                if change > max_change:
                    max_change = change
        
        max_changes[iter] = max_change
        
        # Add relaxation if not converging fast enough
        if iter > 100:
            for i in range(n):
                for j in range(n):
                    V[i,j] = 0.8 * V[i,j] + 0.2 * V_old[i,j]
        
        # Check convergence
        if max_change < tol:
            return V, max_changes[:iter+1], iter+1
            
    return V, max_changes, max_iter

class CableTerminationGenerator:
    def __init__(self, n_points: int = 421, downsample_factor: int = 3):
        """
        Initialize the dataset generator for cable termination problems.
        """
        self.n = n_points
        self.downsample_factor = downsample_factor
        self.dx = 1.0 / (self.n - 1)
        self.dy = 1.0 / (self.n - 1)
        self.downsampled_size = len(np.arange(0, n_points, downsample_factor))
        
        # Create mesh grid
        x = np.linspace(0, 1, self.n)
        y = np.linspace(0, 1, self.n)
        self.X, self.Y = np.meshgrid(x, y)

    def generate_cable_geometry(self, params: Dict) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cable termination geometry with stress control.
        
        params: Dictionary containing:
            - slope: Cone slope
            - conductor_radius: Radius of the conductor
            - stress_control_thickness: Thickness of stress control layer
            - epsilon_insulation: Permittivity of main insulation
            - epsilon_stress_control: Permittivity of stress control material
        """
        epsilon = np.ones((self.n, self.n))  # Air
        voltage_bc_mask = np.zeros((self.n, self.n), dtype=bool)
        voltage_bc_values = np.zeros((self.n, self.n))
        
        # Extract parameters
        slope = params['slope']
        r_conductor = int(params['conductor_radius'] * self.n)
        stress_thickness = int(params['stress_control_thickness'] * self.n)
        eps_insulation = params['epsilon_insulation']
        eps_stress = params['epsilon_stress_control']
        
        # Center coordinates
        center_x = self.n // 2
        
        # Create conductor and stress control regions
        for y in range(self.n):
            # Conductor radius at this height
            if y < self.n // 2:  # Only in lower half
                radius = int(slope * y + r_conductor)
                
                # Conductor region
                x_left = center_x - radius
                x_right = center_x + radius
                if x_left >= 0 and x_right < self.n:
                    epsilon[y, x_left:x_right] = eps_insulation
                    
                    # Add stress control layer
                    stress_left = max(0, x_left - stress_thickness)
                    stress_right = min(self.n, x_right + stress_thickness)
                    epsilon[y, stress_left:x_left] = eps_stress
                    epsilon[y, x_right:stress_right] = eps_stress
                    
                    # Set conductor boundary conditions
                    voltage_bc_mask[y, x_left:x_right] = True
                    voltage_bc_values[y, x_left:x_right] = 1.0  # High voltage
        
        # Ground boundary conditions
        voltage_bc_mask[0, :] = True  # Bottom
        voltage_bc_mask[-1, :] = True  # Top
        voltage_bc_mask[:, 0] = True  # Left
        voltage_bc_mask[:, -1] = True  # Right
        
        return epsilon, (voltage_bc_mask, voltage_bc_values)

    def solve_poisson_fdm(self, epsilon: np.ndarray, voltage_bc: Tuple[np.ndarray, np.ndarray], 
                         max_iter: int = 2000, tol: float = 1e-12) -> np.ndarray:
        """
        Solve 2D Poisson equation for cable termination using FDM.
        """
        V = np.zeros((self.n, self.n))
        V_old = np.zeros((self.n, self.n))
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        
        # Initialize voltage with boundary conditions
        V[voltage_bc[0]] = voltage_bc[1][voltage_bc[0]]
        
        # Call Numba-optimized solver
        V, max_changes, total_iterations = solve_poisson_fdm_numba(
            V, V_old, epsilon, dx2, dy2, self.n, max_iter, tol, voltage_bc
        )
        
        print(f"Converged after {total_iterations} iterations with max change: {max_changes[-1]:.2e}")
        return V

    def generate_sample(self, params: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single sample of cable termination with random parameters if none provided.
        """
        if params is None:
            params = {
                'slope': np.random.uniform(0.3, 0.7),
                'conductor_radius': np.random.uniform(0.05, 0.15),
                'stress_control_thickness': np.random.uniform(0.02, 0.05),
                'epsilon_insulation': np.random.uniform(2.0, 4.0),
                'epsilon_stress_control': np.random.uniform(5.0, 15.0)
            }
        
        epsilon, voltage_bc = self.generate_cable_geometry(params)
        V = self.solve_poisson_fdm(epsilon, voltage_bc)
        
        if self.downsample_factor > 1:
            epsilon = epsilon[::self.downsample_factor, ::self.downsample_factor]
            V = V[::self.downsample_factor, ::self.downsample_factor]
            
        return epsilon, V

    @staticmethod
    def _generate_single_sample_wrapper(args):
        """
        Static wrapper method for parallel processing.
        """
        generator, seed = args
        np.random.seed(seed)
        return generator.generate_sample()


    def generate_and_save_dataset(self, n_samples: int = 1000, 
                            filename: str = 'cable_termination_dataset.mat',
                            n_processes: int = None):
        """
        Generate dataset using parallel processing and save it to a .mat file.
        """
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)
            
        print(f"Generating {n_samples} samples using {n_processes} processes...")
        
        epsilon_samples = np.zeros((n_samples, self.downsampled_size, self.downsampled_size))
        V_samples = np.zeros((n_samples, self.downsampled_size, self.downsampled_size))
        
        # Create seeds for reproducibility
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(n_samples)]
        args = [(self, seed) for seed in seeds]

        # Generate samples using pool.imap
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(self._generate_single_sample_wrapper, args),
                total=n_samples,
                desc="Generating samples"
            ))
            
        for i, (epsilon, V) in enumerate(results):
            epsilon_samples[i] = epsilon
            V_samples[i] = V
            
        print(f"Saving dataset to {filename}...")
        savemat(filename, {
            'input': epsilon_samples,
            'output': V_samples,
            'grid_size': self.downsampled_size,
            'n_samples': n_samples
        })
        print("Dataset saved successfully!")

    def plot_sample(self, epsilon: np.ndarray, V: np.ndarray):
        """
        Plot a sample of dielectric distribution and potential.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(epsilon, cmap='viridis')
        ax1.set_title('Dielectric Distribution')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(V, cmap='viridis')
        ax2.set_title('Potential Distribution')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize generator
    generator = CableTerminationGenerator(n_points=421, downsample_factor=3)
    
    STRR = 'cable_termination_dataset_1000.mat'
    # Generate and save dataset
    generator.generate_and_save_dataset(
        n_samples=1000,
        filename=STRR,
        n_processes=14
    )
    
    # Load and verify the saved dataset
    from scipy.io import loadmat
    data = loadmat(STRR)
    print("\nDataset verification:")
    print(f"Input shape: {data['input'].shape}")
    print(f"Output shape: {data['output'].shape}")
    print(f"Grid size: {data['grid_size'][0,0]}")
    print(f"Number of samples: {data['n_samples'][0,0]}")
    
    # Plot a sample
    generator.plot_sample(data['input'][0], data['output'][0])