"""
rocket_math_utils.py

Utility math functions for rocket simulation, including quaternion conversions
and aerodynamic table interpolation.
"""
import jax
import jax.numpy as jnp
import polars as pl
import elodin as el


def euler_to_quat(angles: jax.Array) -> el.Quaternion:
    """
    Convert Euler angles (degrees) to quaternion.

    Args:
        angles (jax.Array): Array of [roll, pitch, yaw] angles in degrees.

    Returns:
        el.Quaternion: Quaternion representation of the input angles.
    """
    [roll, pitch, yaw] = jnp.deg2rad(angles)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return el.Quaternion(jnp.array([x, y, z, w]))


def aero_interp_table(df: pl.DataFrame) -> jax.Array:
    """
    Interpolate aerodynamic coefficients from a Polars DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing aerodynamic coefficients.

    Returns:
        jax.Array: Interpolated aerodynamic coefficient array.
    """
    coefs = ["CmR", "CA", "CZR"]
    aero = jnp.array(
        [
            [
                df.group_by(["Alphac"], maintain_order=True)
                .agg(pl.col(coefs).min())
                .select(pl.col(coefs))
                .to_numpy()
                for _, df in df.group_by(["Delta"], maintain_order=True)
            ]
            for _, df in df.group_by(["Mach"], maintain_order=True)
        ]
    )
    aero = aero.transpose(3, 0, 1, 2)
    return aero


def to_coord(s: pl.Series, val: jax.Array) -> jax.Array:
    """
    Convert value to coordinate along a series for interpolation.

    Args:
        s (pl.Series): Series to interpolate along.
        val (jax.Array): Value to convert.

    Returns:
        jax.Array: Coordinate value for interpolation.
    """
    s_min = s.min()
    s_max = s.max()
    s_count = len(s.unique())
    return (val - s_min) * (s_count - 1) / jnp.clip(s_max - s_min, 1e-06)
