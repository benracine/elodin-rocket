# Main simulation script for unguided rocket flight using Elodin physics engine.
# This script sets up the world, rocket, and simulation pipeline, and visualizes results.

import elodin as el
import jax.numpy as jnp

from rocket_constants import SIM_TIME_STEP
from rocket_math_utils import euler_to_quat
from rocket_model import Rocket
from rocket_physics import mach, angle_of_attack, aero_coefs, aero_forces, apply_aero_forces, gravity, thrust, apply_thrust
from rocket_types import Thrust

# Create the simulation world
w = el.World()

# Define the target position (meters)
target = jnp.array([400.0, 0.0, 0.0])  # 400 meters is realistic for AeroTech M685W

# Define the rocket launch position (meters)
# 0.5 meters above ground simulates a typical launch rail or pad
launch_pos = jnp.array([0.0, 0.0, 0.5])

# Calculate the direction vector from launch to target and normalize it
direction = target - launch_pos
direction = direction / jnp.linalg.norm(direction)

# Set pitch angle for maximum range (optimized via range_optimizer.py)
pitch = 35.0  # degrees above horizontal (optimized for maximum range)
# Calculate yaw angle to aim toward the target in XY plane
yaw = jnp.rad2deg(jnp.arctan2(direction[1], direction[0]))
aim_euler = jnp.array([0.0, pitch, yaw])  # [Roll, Pitch, Yaw] in degrees

# Print out the aim angles for reference
print(f"Aim angles: Pitch = {pitch:.2f} deg, Yaw = {yaw:.2f} deg")

# Spawn the rocket entity with initial position and orientation

# Define rocket mass properties
dry_mass = 3.0        # kg, mass of rocket without payload
payload_mass = 5.0    # kg, mass of payload
total_mass = dry_mass + payload_mass  # kg, total mass

# The rocket's total mass is set using the sum of dry and payload mass
rocket = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                angular=euler_to_quat(aim_euler),  # Aim at target
                linear=launch_pos,                  # Initial position
            ),
            inertia=el.SpatialInertia(total_mass, jnp.array([0.1, 1.0, 1.0])),  # Use total_mass variable
        ),
        Rocket(),
        w.glb("https://storage.googleapis.com/elodin-assets/rocket.glb"),  # 3D model for visualization
    ],
    name="Rocket",
)

# Visualize rocket trajectory as a thick line
w.spawn(el.Line3d(rocket, line_width=11.0))


# Set up simulation dashboard with 3D viewport and graphs
w.spawn(
    el.Panel.hsplit(
        el.Panel.vsplit(
            el.Panel.viewport(
                track_entity=rocket,
                track_rotation=False,
                pos=[5.0, 0.0, 1.0],
                looking_at=[0.0, 0.0, 0.0],
                show_grid=True,
            ),
        ),
        el.Panel.vsplit(
            el.Panel.graph(el.GraphEntity(rocket, Thrust)),  # Thrust over time
            el.Panel.graph(
                el.GraphEntity(
                    rocket,
                    el.WorldPos  # Rocket position over time
                )
            ),
        ),
        active=True,
    )
)


# Define the simulation pipeline
# non_effectors: state update functions (physics, aerodynamics, thrust)
# effectors: force application functions (gravity, thrust, aerodynamics)
non_effectors = (
    mach
    | angle_of_attack
    | aero_coefs
    | aero_forces
    | thrust
)
effectors = gravity | apply_thrust | apply_aero_forces
sys = non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)

# Run the simulation
w.run(sys, sim_time_step=SIM_TIME_STEP)
