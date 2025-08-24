#!/usr/bin/env python3
"""
Quick validation script to compare the old vs optimized pitch angles.
"""

from range_optimizer import calculate_rocket_trajectory

def main():
    print("Rocket Launch Pitch Angle Optimization Results")
    print("=" * 50)
    
    # Original pitch angle from rocket_sim.py
    original_pitch = 8.0
    original_range = calculate_rocket_trajectory(original_pitch)
    
    # Optimized pitch angle
    optimal_pitch = 35.0
    optimal_range = calculate_rocket_trajectory(optimal_pitch)
    
    # Calculate improvement
    improvement = ((optimal_range - original_range) / original_range) * 100
    
    print(f"Original pitch angle: {original_pitch}°")
    print(f"Original range: {original_range:.1f} m")
    print()
    print(f"Optimized pitch angle: {optimal_pitch}°") 
    print(f"Optimized range: {optimal_range:.1f} m")
    print()
    print(f"Range improvement: {improvement:.1f}%")
    print(f"Additional distance: {optimal_range - original_range:.1f} m")
    
    print(f"\nThe rocket simulation (rocket_sim.py) has been updated to use")
    print(f"the optimal pitch angle of {optimal_pitch}° for maximum range.")

if __name__ == "__main__":
    main()