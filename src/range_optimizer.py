#!/usr/bin/env python3
"""
Range optimization script for rocket simulation.

This script analyzes different pitch angles to find the angle that produces 
the maximum range using a physics-based model derived from the simulation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Tuple, List
from thrust_curve import thrust_curve


def calculate_rocket_trajectory(pitch_angle: float) -> float:
    """
    Calculate the range for a given pitch angle using rocket physics.
    
    This uses the actual rocket parameters from the simulation:
    - Motor thrust curve (AeroTech M685W)
    - Rocket mass (8 kg total)
    - Aerodynamic drag approximation
    - Gravity
    
    Args:
        pitch_angle: Launch pitch angle in degrees
        
    Returns:
        Range achieved by the rocket (horizontal distance when it lands)
    """
    # Rocket parameters from simulation
    dry_mass = 3.0      # kg
    payload_mass = 5.0  # kg
    total_mass = dry_mass + payload_mass  # 8 kg
    
    # Convert pitch angle to radians
    angle_rad = np.radians(pitch_angle)
    
    # Thrust curve data
    time_points = np.array(thrust_curve["time"])
    thrust_points = np.array(thrust_curve["thrust"])
    
    # Simulation parameters
    dt = 0.01  # time step in seconds
    g = 9.81   # gravity
    launch_height = 0.5  # meters
    
    # Aerodynamic parameters (simplified)
    drag_coefficient = 0.5  # estimated from aerodynamics table
    reference_area = 0.002489130  # m^2 from rocket_constants
    air_density = 1.225  # kg/m^3 at sea level
    
    # Initial conditions
    x = 0.0
    y = launch_height
    vx = 0.0
    vy = 0.0
    t = 0.0
    
    trajectory_x = [x]
    trajectory_y = [y]
    trajectory_t = [t]
    
    # Simulation loop
    while y >= 0.0 and t < 120.0:  # until landing or max time
        # Current velocity magnitude
        v_mag = np.sqrt(vx*vx + vy*vy)
        
        # Thrust force (interpolate from curve)
        if t <= time_points[-1]:
            current_thrust = np.interp(t, time_points, thrust_points)
        else:
            current_thrust = 0.0
            
        # Thrust components (in launch direction)
        thrust_x = current_thrust * np.cos(angle_rad)
        thrust_y = current_thrust * np.sin(angle_rad)
        
        # Drag force (opposite to velocity direction)
        if v_mag > 0:
            drag_force = 0.5 * air_density * drag_coefficient * reference_area * v_mag * v_mag
            drag_x = -drag_force * (vx / v_mag)
            drag_y = -drag_force * (vy / v_mag)
        else:
            drag_x = drag_y = 0.0
        
        # Total forces
        fx = thrust_x + drag_x
        fy = thrust_y + drag_y - total_mass * g
        
        # Acceleration
        ax = fx / total_mass
        ay = fy / total_mass
        
        # Update velocity and position (Euler integration)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        
        # Store trajectory
        if len(trajectory_t) % 10 == 0:  # Store every 10th point to reduce memory
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_t.append(t)
    
    # Return the final horizontal distance (range)
    return x


def optimize_pitch_angle(pitch_range: Tuple[float, float] = (5.0, 85.0), 
                        num_points: int = 17) -> Tuple[float, float, List[float], List[float]]:
    """
    Find the optimal pitch angle for maximum range.
    
    Args:
        pitch_range: Tuple of (min_pitch, max_pitch) in degrees
        num_points: Number of pitch angles to test
        
    Returns:
        Tuple of (optimal_pitch, max_range, pitch_angles_tested, ranges_achieved)
    """
    print(f"Testing {num_points} pitch angles from {pitch_range[0]}° to {pitch_range[1]}°")
    
    # Generate pitch angles to test
    pitch_angles = np.linspace(pitch_range[0], pitch_range[1], num_points)
    ranges = []
    
    # Test each pitch angle
    for i, pitch in enumerate(pitch_angles):
        print(f"Testing pitch angle {pitch:.1f}° ({i+1}/{num_points})", end=" ")
        try:
            range_achieved = calculate_rocket_trajectory(pitch)
            ranges.append(range_achieved)
            print(f"-> Range: {range_achieved:.1f} m")
        except Exception as e:
            print(f"-> Error: {e}")
            ranges.append(0.0)
    
    # Find optimal pitch angle
    max_range_idx = np.argmax(ranges)
    optimal_pitch = pitch_angles[max_range_idx]
    max_range = ranges[max_range_idx]
    
    return optimal_pitch, max_range, list(pitch_angles), ranges


def plot_results(pitch_angles: List[float], ranges: List[float], optimal_pitch: float, max_range: float):
    """Plot the pitch angle vs range results."""
    plt.figure(figsize=(10, 6))
    plt.plot(pitch_angles, ranges, 'b-o', linewidth=2, markersize=6)
    plt.axvline(optimal_pitch, color='r', linestyle='--', alpha=0.7, 
                label=f'Optimal: {optimal_pitch:.1f}° ({max_range:.1f} m)')
    plt.xlabel('Pitch Angle (degrees)')
    plt.ylabel('Range (meters)')
    plt.title('Rocket Range vs Launch Pitch Angle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/tmp/pitch_optimization_results.png', dpi=300, bbox_inches='tight')
    print("Results plot saved to /tmp/pitch_optimization_results.png")
    
    plt.show()


def main():
    """Main function to run the pitch angle optimization."""
    print("Rocket Range Optimization - Finding Optimal Launch Pitch Angle")
    print("=" * 60)
    
    # Run optimization
    optimal_pitch, max_range, pitch_angles, ranges = optimize_pitch_angle()
    
    print(f"\nOptimization Results:")
    print(f"Optimal pitch angle: {optimal_pitch:.1f}°")
    print(f"Maximum range achieved: {max_range:.1f} m")
    print(f"Current simulation uses: 8.0°")
    
    # Calculate current performance
    current_range = calculate_rocket_trajectory(8.0)
    improvement = ((max_range - current_range) / current_range) * 100 if current_range > 0 else 0
    print(f"Current 8.0° range: {current_range:.1f} m")
    print(f"Range improvement: {improvement:.1f}% over current 8.0° setting")
    
    # Plot results
    plot_results(pitch_angles, ranges, optimal_pitch, max_range)
    
    # Print detailed results
    print(f"\nDetailed Results:")
    print("Pitch (°) | Range (m)")
    print("-" * 20)
    for pitch, range_val in zip(pitch_angles, ranges):
        marker = " *" if abs(pitch - optimal_pitch) < 0.1 else ""
        print(f"{pitch:8.1f} | {range_val:8.1f}{marker}")
    
    print(f"\nTo use the optimal pitch angle, modify rocket_sim.py line 28:")
    print(f"Change: pitch = 8.0  # degrees above horizontal")  
    print(f"To:     pitch = {optimal_pitch:.1f}  # degrees above horizontal (optimized)")


if __name__ == "__main__":
    main()