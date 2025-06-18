
import matplotlib.pyplot as plt
import numpy as np


class RealTimeTrailVisualizer:
    def __init__(self):
        self.trail_points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(0.2, 0.8)
        self.ax.set_ylim(-0.3, 0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Robot Drawing Trail - "Hello"')
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Drawing path')
        self.current_point, = self.ax.plot([], [], 'ro', markersize=8, label='Current position')
        self.ax.legend()
        plt.ion()
        plt.show()
    
    def add_point(self, pos, is_drawing):
        if is_drawing:
            self.trail_points.append([pos[0], pos[1]])
            
            if len(self.trail_points) > 1:
                trail_array = np.array(self.trail_points)
                self.line.set_data(trail_array[:, 0], trail_array[:, 1])
        
        # Always update current position
        self.current_point.set_data([pos[0]], [pos[1]])
        
        # Update plot every 10 points to avoid slowdown
        if len(self.trail_points) % 10 == 0:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
