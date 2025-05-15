"""This example compares custom rotation matrix implementations with scipy's rotation functions.
It includes functions to rotate points around the x, y, and z axes using both custom and scipy methods.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

### Custom Rotation Functions ###
def rotate_x(theta):
    """
    Create a rotation matrix for a rotation around the x-axis.
    
    Parameters:
    theta (float): The angle of rotation in degrees.
    
    Returns:
    numpy.ndarray: The rotation matrix.
    """
    theta_rad = np.radians(theta)
    
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])
    return rot

def rotate_y(theta):
    """
    Create a rotation matrix for a rotation around the y-axis.
    
    Parameters:
    theta (float): The angle of rotation in degrees.
    
    Returns:
    numpy.ndarray: The rotation matrix.
    """
    theta_rad = np.radians(theta)
    
    rot = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    return rot

def rotate_z(theta):
    """
    Create a rotation matrix for a rotation around the z-axis.
    
    Parameters:
    theta (float): The angle of rotation in degrees.
    
    Returns:
    numpy.ndarray: The rotation matrix.
    """
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    
    rot = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    return rot

def rotate_point(point, theta, axis):
    """
    Rotate a point around a specified axis using custom implementation.
    
    Parameters:
    point (numpy.ndarray): The point to be rotated.
    theta (float): The angle of rotation in degrees.
    axis (str): The axis of rotation ('x', 'y', or 'z').
    
    Returns:
    numpy.ndarray: The rotated point.
    """
    if axis == 'x':
        rotation_matrix = rotate_x(theta)
    elif axis == 'y':
        rotation_matrix = rotate_y(theta)
    elif axis == 'z':
        rotation_matrix = rotate_z(theta)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return np.dot(rotation_matrix, point) 


### Scipy Rotation Functions ###
def rotate_point_scipy(point, theta, axis):
    """
    Rotate a point around a specified axis using scipy.
    
    Parameters:
    point (numpy.ndarray): The point to be rotated.
    theta (float): The angle of rotation in degrees.
    axis (str): The axis of rotation ('x', 'y', or 'z').
    
    Returns:
    numpy.ndarray: The rotated point.
    """
    if axis == 'x':
        r = R.from_euler('x', theta, degrees=True)
    elif axis == 'y':
        r = R.from_euler('y', theta, degrees=True)
    elif axis == 'z':
        r = R.from_euler('z', theta, degrees=True)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return r.apply(point)


### Comparison and Animation Functions ###
def compare_rotations(theta=90, axis='z'):
    """
    Compare the rotation of a point using custom and scipy methods.
    
    Parameters:
    theta (float): The angle of rotation in degrees.
    axis (str): The axis of rotation ('x', 'y', or 'z').
    
    Returns:
    None
    """
    point = np.array([1, 0, 0])

    rotated_point_custom = rotate_point(point, theta, axis)
    rotated_point_scipy = rotate_point_scipy(point, theta, axis)
    
    print(f"Rotation around {axis}-axis by {theta} degrees:")
    print("Original Point:", point)
    print("Rotated Point (Custom):", rotated_point_custom)
    print("Rotated Point (Scipy):", rotated_point_scipy)
    
    if np.allclose(rotated_point_custom, rotated_point_scipy, atol=1e-15):
        print("The results are numerically equivalent (within tolerance).")
    else:
        print("There are small numerical differences (floating-point precision artifacts).")
        print("Difference:", rotated_point_scipy - rotated_point_custom)


def animate_rotation_comparison(axis='z', duration=10, fps=30):
    """
    Animate and compare both rotation methods side by side with a static view.
    
    Parameters:
    axis (str): The axis of rotation ('x', 'y', or 'z').
    duration (float): Animation duration in seconds.
    fps (int): Frames per second.
    
    Returns:
    matplotlib.animation.FuncAnimation: The animation object.
    """
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax1.set_title('Custom Rotation')
    ax2.set_title('SciPy Rotation')
    
    if axis == 'x':
        point = np.array([0,1, 0]) # Point on Y-axis
    elif axis == 'y':
        point = np.array([1, 0, 0]) # Point on X-axis
    elif axis == 'z':
        point = np.array([1, 0, 0]) # Point on X-axis


    total_frames = int(duration * fps)
    angles = np.linspace(0, 360, total_frames)
    
    if axis == 'x':
        elev, azim = 20, -80 
    elif axis == 'y':
        elev, azim = 20, 10   
    else:  # z-axis
        elev, azim = 30, -45  
    
    for ax in [ax1, ax2]:
        ax.view_init(elev=elev, azim=azim)
    
    original_vectors = []
    original_vectors.append(ax1.quiver(0, 0, 0, point[0], point[1], point[2], color='blue', label='Original', arrow_length_ratio=0.1))
    original_vectors.append(ax2.quiver(0, 0, 0, point[0], point[1], point[2], color='blue', label='Original', arrow_length_ratio=0.1))
    
    rotated_vectors = []
    rotated_vectors.append(ax1.quiver(0, 0, 0, 0, 0, 0, color='red', label='Custom Rotation', arrow_length_ratio=0.1))
    rotated_vectors.append(ax2.quiver(0, 0, 0, 0, 0, 0, color='red', label='SciPy Rotation', arrow_length_ratio=0.1))
    
    length = 1.2
    axes_colors = ['#aa0000', '#00aa00', '#0000aa']  # Red, Green, Blue for X, Y, Z
    axis_labels = ['X', 'Y', 'Z']
    
    for ax in [ax1, ax2]:
        ax.quiver(0, 0, 0, length, 0, 0, color=axes_colors[0], arrow_length_ratio=0.05, linestyle='dashed', alpha=0.7)
        ax.quiver(0, 0, 0, 0, length, 0, color=axes_colors[1], arrow_length_ratio=0.05, linestyle='dashed', alpha=0.7)
        ax.quiver(0, 0, 0, 0, 0, length, color=axes_colors[2], arrow_length_ratio=0.05, linestyle='dashed', alpha=0.7)
        
        ax.text(length*1.1, 0, 0, axis_labels[0], color=axes_colors[0])
        ax.text(0, length*1.1, 0, axis_labels[1], color=axes_colors[1])
        ax.text(0, 0, length*1.1, axis_labels[2], color=axes_colors[2])
        
        # Highlight the rotation axis
        if axis == 'x':
            ax.quiver(0, 0, 0, length, 0, 0, color=axes_colors[0], arrow_length_ratio=0.05, alpha=1.0, linewidth=2)
        elif axis == 'y':
            ax.quiver(0, 0, 0, 0, length, 0, color=axes_colors[1], arrow_length_ratio=0.05, alpha=1.0, linewidth=2)
        else:  # z
            ax.quiver(0, 0, 0, 0, 0, length, color=axes_colors[2], arrow_length_ratio=0.05, alpha=1.0, linewidth=2)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        
        ax.grid(True, alpha=0.3, linestyle='--')
    
    difference_text = fig.text(0.5, 0.02, '', ha='center', va='center', fontsize=10)
    
    fig.suptitle(f'Rotation around {axis}-axis', fontsize=16)
    
    ax1.legend(loc='upper right')
    
 
    def update(frame):
        angle = angles[frame]
        
        fig.suptitle(f'Rotation around {axis.upper()}-axis (Angle: {angle:.1f}°)', fontsize=16)
        
        rotated_custom = rotate_point(point, angle, axis)
        rotated_scipy = rotate_point_scipy(point, angle, axis)
        
        rotated_vectors[0].remove()
        rotated_vectors[1].remove()
        
        rotated_vectors[0] = ax1.quiver(0, 0, 0, rotated_custom[0], rotated_custom[1], rotated_custom[2], 
                                         color='red', label='Custom Rotation', arrow_length_ratio=0.1)
        rotated_vectors[1] = ax2.quiver(0, 0, 0, rotated_scipy[0], rotated_scipy[1], rotated_scipy[2], 
                                         color='red', label='SciPy Rotation', arrow_length_ratio=0.1)
        
        diff = np.linalg.norm(rotated_scipy - rotated_custom)
        difference_text.set_text(f'Numerical difference: {diff:.2e}')
        
        # The view remains static - no view_init calls
        
        return rotated_vectors + [difference_text]
    
    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
    plt.tight_layout()
    
    return ani


def main():
    """
    Main function to run the rotation comparisons and animation.
    
    Returns:
    None
    """
    args = argparse.ArgumentParser(description="Compare custom and scipy rotations.")
    args.add_argument('--axis', type=str, choices=['x', 'y', 'z'], default='z', help="Axis of rotation (default: z)")
    args.add_argument('--save', action='store_true', default=False, help="Save the animation as a video file")
    args.add_argument('--save_dir', type=str, default='../visualizations', help="Directory to save the animation")
    args = args.parse_args()
    axis_to_animate = args.axis
    for axis in ['x', 'y', 'z']:
        compare_rotations(90, axis)
        print()
    
    ani = animate_rotation_comparison(axis=axis_to_animate)
    if args.save:
        import os
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ani.save(os.path.join(args.save_dir, f'rotation_{axis_to_animate}.mp4'), writer='ffmpeg', fps=30)    
    plt.show()

if __name__ == "__main__":
    main()