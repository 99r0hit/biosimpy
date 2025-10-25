import numpy as np
import matplotlib.pyplot as plt

class MicrofluidicSimulator:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        
    def simulate_y_channel(self, flow_rate=1.0, viscosity=0.01, steps=200):
        """Simulate flow in a Y-shaped microchannel"""
        # Initialize fields
        velocity_x = np.zeros((self.height, self.width))
        velocity_y = np.zeros((self.height, self.width))
        concentration = np.zeros((self.height, self.width))
        
        # Create Y-channel mask
        mask = self._create_y_channel_mask()
        
        # Set inlets
        left_inlet1 = (self.height//2 - 15, 0)
        left_inlet2 = (self.height//2 + 15, 0)
        right_outlet = (self.height//2, self.width-1)
        
        # Simulation loop
        for step in range(steps):
            # Apply boundary conditions
            velocity_x[left_inlet1[0], left_inlet1[1]] = flow_rate
            velocity_x[left_inlet2[0], left_inlet2[1]] = flow_rate
            concentration[left_inlet1[0], left_inlet1[1]] = 0.8  # Different concentrations
            concentration[left_inlet2[0], left_inlet2[1]] = 0.4
            
            # Simple diffusion and advection
            concentration = self._diffuse(concentration, mask, 0.1)
            concentration = self._advect(concentration, velocity_x, velocity_y, mask)
            
            # Ensure concentration stays in bounds
            concentration = np.clip(concentration, 0, 1)
            
        return velocity_x, velocity_y, concentration, mask
    
    def _create_y_channel_mask(self):
        """Create a Y-shaped channel mask"""
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Main channel (horizontal)
        center_y = self.height // 2
        mask[center_y-5:center_y+5, :] = True
        
        # Left inlets (Y shape)
        for i in range(20):
            y1 = center_y - 5 - i
            y2 = center_y + 5 + i
            if 0 <= y1 < self.height:
                mask[y1, :self.width//3] = True
            if 0 <= y2 < self.height:
                mask[y2, :self.width//3] = True
                
        return mask
    
    def _diffuse(self, concentration, mask, diffusion_coeff):
        """Simple diffusion implementation"""
        diffused = concentration.copy()
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if mask[i, j]:
                    laplacian = (concentration[i-1, j] + concentration[i+1, j] +
                                concentration[i, j-1] + concentration[i, j+1] -
                                4 * concentration[i, j])
                    diffused[i, j] += diffusion_coeff * laplacian
        return diffused
    
    def _advect(self, concentration, vel_x, vel_y, mask):
        """Simple advection implementation"""
        advected = concentration.copy()
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if mask[i, j] and vel_x[i, j] > 0:
                    # Simple upwind scheme
                    advected[i, j] = 0.8 * concentration[i, j-1] + 0.2 * concentration[i, j]
        return advected
    
    def visualize_flow(self, velocity_x, velocity_y, concentration, mask):
        """Create flow visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Velocity magnitude
        velocity_mag = np.sqrt(velocity_x**2 + velocity_y**2)
        im1 = ax1.imshow(velocity_mag, cmap='viridis', alpha=0.7)
        ax1.imshow(mask, cmap='gray', alpha=0.3)
        ax1.set_title('Velocity Magnitude')
        plt.colorbar(im1, ax=ax1)
        
        # Concentration
        im2 = ax2.imshow(concentration, cmap='plasma', alpha=0.7)
        ax2.imshow(mask, cmap='gray', alpha=0.3)
        ax2.set_title('Concentration Distribution')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        return fig
