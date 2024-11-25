import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple

def animate_optimization(history_positions: List[List[np.ndarray]], bounds: Tuple[float, float], optimum: np.ndarray):
    """
    Create an animation of the whale optimization algorithm with a background gradient.
    
    Args:
        history_positions: List of whale positions at each iteration.
        bounds: Tuple of (min_bound, max_bound) for the plot axes.
        optimum: The optimum position (x, y) to highlight.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_title("Whale Optimization")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    # Create a background gradient
    x = np.linspace(bounds[0], bounds[1], 500)
    y = np.linspace(bounds[0], bounds[1], 500)
    X, Y = np.meshgrid(x, y)
    
    Z = np.sqrt((X - optimum[0])**2 + (Y - optimum[1])**2)
    
    # Plot the background as a contour plot
    background = ax.contourf(X, Y, Z, levels=100, cmap="viridis", alpha=0.7)
    plt.colorbar(background, ax=ax, label="Distance from Optimum")

    # Scatter plot for whale positions
    scatter = ax.scatter([], [], c="white", s=60, label="Whales")
    ax.legend()

    def update(frame):
        # Get positions for the current frame
        positions = history_positions[frame]
        
        # Convert positions to (x, y) coordinates
        coords = np.array(positions)
        
        # Update the scatter plot
        scatter.set_offsets(coords)
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(history_positions), blit=True, repeat=False)
    plt.show()