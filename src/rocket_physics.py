"""
aero.py

Aerodynamics-related functions for rocket simulation.
This module provides functions to compute Mach number, dynamic pressure,
angle of attack, aerodynamic coefficients, and aerodynamic forces for a rocket.
All functions are designed to be modular and reusable in the simulation pipeline.
"""
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax.scipy.ndimage import map_coordinates
import polars as pl

import elodin as el

from rocket_math_utils import aero_interp_table, to_coord
from rocket_model import Wind, Mach, DynamicPressure, AngleOfAttack, AeroCoefs, AeroForce, CenterOfGravity, Motor, Thrust
from rocket_constants import thrust_vector_body_frame, a_ref, l_ref, xmc
from thrust_curve import thrust_curve


# Aerodynamic coefficient table
aero_df = pl.from_dict({
    'Mach': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'Alphac': [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0],
    'Delta': [-40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0],
    'CmR': [-5.997, -6.905, -8.235, -10.83, -5.315, -6.008, -5.918, -5.714, 0.0, 1.313, 2.335, 0.4163, 5.315, 3.642, 2.977, 1.061, 5.997, 5.372, 4.191, 1.882, -7.269, -8.373, -9.873, -12.93, -6.323, -7.255, -7.14, -6.846, 0.0, 1.486, 2.681, 0.445, 6.323, 4.263, 3.463, 1.222, 7.269, 6.396, 4.963, 2.27, -11.53, -12.49, -13.88, -15.71, -9.056, -8.891, -8.448, -8.155, 0.0, 1.921, 3.144, 1.169, 9.056, 8.419, 7.126, 4.228, 11.53, 10.14, 8.19, 4.94],
    'CA': [1.121, 1.028, 0.9495, 0.9803, 0.6405, 0.5852, 0.4342, 0.217, 0.2942, 0.2873, 0.2591, 0.2032, 0.6405, 0.5988, 0.635, 0.6333, 1.121, 1.215, 1.246, 1.267, 1.242, 1.137, 1.051, 1.095, 0.6902, 0.6278, 0.4588, 0.2184, 0.2924, 0.2856, 0.2577, 0.2025, 0.6902, 0.6434, 0.6895, 0.6967, 1.242, 1.351, 1.392, 1.425, 1.851, 1.747, 1.621, 1.48, 0.9888, 0.8509, 0.658, 0.4269, 0.448, 0.4446, 0.4345, 0.418, 0.9888, 1.06, 1.111, 1.154, 1.851, 1.961, 2.03, 2.098],
    'CZR': [-1.092, -0.3878, 0.3984, 1.141, -1.141, -0.4069, 0.7324, 2.176, 0.0, 1.061, 2.368, 3.494, 1.141, 1.561, 2.483, 3.64, 1.092, 1.789, 2.577, 3.68, -1.191, -0.4161, 0.4355, 1.252, -1.274, -0.4526, 0.8073, 2.408, 0.0, 1.178, 2.63, 3.88, 1.274, 1.736, 2.755, 4.043, 1.191, 1.973, 2.844, 4.07, -1.609, -0.8494, 0.1373, 1.323, -1.639, -0.5395, 0.9159, 2.704, 0.0, 1.304, 2.894, 4.443, 1.639, 2.532, 3.576, 4.981, 1.609, 2.483, 3.481, 4.811]
})  # fmt: skip

@el.map
def mach(p: el.WorldPos, v: el.WorldVel, w: Wind) -> tuple[Mach, DynamicPressure]:
    """
    Calculate Mach number and dynamic pressure for the rocket.

    Args:
        p (el.WorldPos): Rocket position and orientation in world frame.
        v (el.WorldVel): Rocket velocity in world frame.
        w: Wind vector in world frame.

    Returns:
        tuple: (Mach number, dynamic pressure)
    """
    # Standard atmosphere model (layered)
    atmosphere = {
        "h": jnp.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]),
        "T": jnp.array([15.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.2]),
        "p": jnp.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.0]),
        "d": jnp.array([1.225, 0.3639, 0.0880, 0.0132, 0.0014, 0.0009, 0.0001, 0.0])
    }
    altitude = p.linear()[2]  # Z component is altitude
    temperature = jnp.interp(altitude, atmosphere["h"], atmosphere["T"]) + 273.15  # Kelvin
    density = jnp.interp(altitude, atmosphere["h"], atmosphere["d"])
    specific_heat_ratio = 1.4
    specific_gas_constant = 287.05
    speed_of_sound = jnp.sqrt(specific_heat_ratio * specific_gas_constant * temperature)
    local_flow_velocity = la.norm(v.linear() - w)
    mach = local_flow_velocity / speed_of_sound
    dynamic_pressure = 0.5 * density * local_flow_velocity**2
    dynamic_pressure = jnp.clip(dynamic_pressure, 1e-6)  # Avoid zero/negative pressure
    return mach, dynamic_pressure


@el.map
def angle_of_attack(p: el.WorldPos, v: el.WorldVel, w: Wind) -> AngleOfAttack:
    """
    Calculate the angle of attack (degrees) between the rocket's axis and the freestream velocity.

    Args:
        p (el.WorldPos): Rocket position and orientation in world frame.
        v (el.WorldVel): Rocket velocity in world frame.
        w: Wind vector in world frame.
        thrust_vector_body_frame: Reference thrust direction in body frame.

    Returns:
        float: Angle of attack in degrees.
    """
    # Transform freestream velocity to body frame
    u = p.angular().inverse() @ (v.linear() - w)
    # Compute angle between velocity and thrust axis
    angle_of_attack = jnp.dot(u, thrust_vector_body_frame) / jnp.clip(la.norm(u), 1e-6)
    angle_of_attack = jnp.rad2deg(jnp.arccos(angle_of_attack)) * -jnp.sign(u[2])
    return angle_of_attack


@el.map
def aero_coefs(mach: Mach, angle_of_attack: AngleOfAttack) -> AeroCoefs:
    """
    Interpolate aerodynamic coefficients for the rocket from a lookup table.

    Args:
        mach (float): Mach number.
        angle_of_attack (float): Angle of attack in degrees.
        aero_df: DataFrame containing aerodynamic coefficient tables.

    Returns:
        jnp.ndarray: Array of aerodynamic coefficients [Cl, CnR, CmR, CA, CZR, CYR].
    """
    aero = aero_interp_table(aero_df)
    # Determine sign for angle of attack (for negative interpolation)
    aoa_sign = jax.lax.cond(
        jnp.abs(angle_of_attack) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(angle_of_attack),
        None
    )
    # Interpolate coefficients using Mach and Alphac (AOA)
    coords = [
        to_coord(aero_df["Mach"], mach),
        to_coord(aero_df["Delta"], 0.0),  # No fin deflection
        to_coord(aero_df["Alphac"], jnp.abs(angle_of_attack)),
    ]
    coefs = jnp.array([
        map_coordinates(coef, coords, 1, mode="nearest") for coef in aero
    ])
    # Output format matches simulation expectations
    coefs = jnp.array([
        0.0,                # Cl (roll moment, unused)
        0.0,                # CnR (yaw moment, unused)
        coefs[0] * aoa_sign,  # CmR (pitch moment)
        coefs[1],             # CA (axial force)
        coefs[2] * aoa_sign,  # CZR (normal force)
        0.0,                # CYR (side force, unused)
    ])
    return coefs


@el.map
def aero_forces(aero_coefs: AeroCoefs, xcg: CenterOfGravity, q: DynamicPressure) -> AeroForce:
    """
    Calculate aerodynamic forces and torques acting on the rocket.

    Args:
        aero_coefs (jnp.ndarray): Aerodynamic coefficients [Cl, CnR, CmR, CA, CZR, CYR].
        xcg (float): Center of gravity position.
        q (float): Dynamic pressure.
        xmc (float): Reference moment center position.
        l_ref (float): Reference length.
        a_ref (float): Reference area.

    Returns:
        el.SpatialForce: Combined linear and torque aerodynamic force.
    """
    Cl, CnR, CmR, CA, CZR, CYR = aero_coefs
    # Shift moments from moment center to center of gravity
    CmR = CmR - CZR * (xcg - xmc) / l_ref
    CnR = CnR - CYR * (xcg - xmc) / l_ref
    # Compute linear and torque forces
    f_aero_linear = jnp.array([CA, CYR, CZR]) * q * a_ref
    f_aero_torque = jnp.array([Cl, -CmR, CnR]) * q * a_ref * l_ref
    f_aero = el.SpatialForce(linear=f_aero_linear, torque=f_aero_torque)
    return f_aero


@el.map
def apply_aero_forces(p: el.WorldPos, f_aero: AeroForce, f: el.Force) -> el.Force:
    """
    Apply aerodynamic forces to the rocket in the world frame.

    Args:
        p (el.WorldPos): Rocket position and orientation in world frame.
        f_aero (el.SpatialForce): Aerodynamic force and torque in body frame.
        f (el.Force): Existing force to be updated.

    Returns:
        el.Force: Updated force including aerodynamic effects.
    """
    # Transform aerodynamic force from body to world frame and add to total force
    return f + p.angular() @ f_aero


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())

@el.system
def thrust(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    q: el.Query[Motor],
) -> el.Query[Thrust]:
    t = tick[0] * dt[0]
    time = jnp.array(thrust_curve["time"])
    thrust = jnp.array(thrust_curve["thrust"])
    f_t = jnp.interp(t, time, thrust)
    return q.map(Thrust, lambda _: f_t)

@el.map
def apply_thrust(thrust: Thrust, f: el.Force, p: el.WorldPos) -> el.Force:
    return f + el.SpatialForce(linear=p.angular() @ thrust_vector_body_frame * thrust)
