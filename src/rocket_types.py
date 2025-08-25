"""
rocket_types.py

Type definitions for rocket simulation components, using Elodin and JAX.
"""

import typing as ty
import jax

import elodin as el
from rocket_constants import lp_buffer_size

AccelSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 101},
    ),
]
"""
2D array for pitch and yaw acceleration setpoints.
Shape: (2,)
Units: m/s²
Usage: Desired angular acceleration setpoints for pitch and yaw axes, used by the control system.
"""

AccelSetpointSmooth = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint_smooth",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 100},
    ),
]
"""
Smoothed 2D array for pitch and yaw acceleration setpoints.
Shape: (2,)
Units: m/s²
Usage: Filtered version of AccelSetpoint for smoother control response.
"""

AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "Cl,CnR,CmR,CA,CZR,CYR"},
    ),
]
"""
6D aerodynamic coefficients array.
Shape: (6,)
Elements: Cl (lift), CnR (normal roll), CmR (moment roll), CA (axial), CZR (side force), CYR (yaw force)
Units: Dimensionless
Usage: Used in aerodynamic force and moment calculations.
"""

AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "aero_force",
        el.ComponentType.SpatialMotionF64,
        metadata={"element_names": "τx,τy,τz,x,y,z"},
    ),
]
"""
Spatial force from aerodynamic effects.
Shape: (6,)
Elements: [torque_x, torque_y, torque_z, force_x, force_y, force_z]
Units: N·m (torque), N (force)
Usage: Resultant aerodynamic force and torque applied to the rocket body.
"""

AngleOfAttack = ty.Annotated[
    jax.Array,
    el.Component("angle_of_attack", el.ComponentType.F64),
]
"""
Scalar angle of attack value.
Shape: (1,)
Units: degrees
Usage: Angle between rocket's velocity vector and its longitudinal axis.
"""

CenterOfGravity = ty.Annotated[
    jax.Array,
    el.Component("center_of_gravity", el.ComponentType.F64),
]
"""
Scalar center of gravity position.
Shape: (1,)
Units: meters
Usage: Longitudinal position of the rocket's center of gravity.
"""

DynamicPressure = ty.Annotated[
    jax.Array,
    el.Component("dynamic_pressure", el.ComponentType.F64),
]
"""
Scalar dynamic pressure value.
Shape: (1,)
Units: Pascals (Pa)
Usage: Used in aerodynamic force calculations; q = 0.5 * density * velocity².
"""

FinControl = ty.Annotated[
    jax.Array,
    el.Component("fin_control", el.ComponentType.F64),
]
"""
Scalar fin control value.
Shape: (1,)
Units: degrees
Usage: Commanded deflection for rocket control fins.
"""

FinDeflect = ty.Annotated[
    jax.Array,
    el.Component("fin_deflect", el.ComponentType.F64),
]
"""
Scalar fin deflection value.
Shape: (1,)
Units: degrees
Usage: Actual deflection angle of rocket control fins.
"""

Mach = ty.Annotated[
    jax.Array,
    el.Component("mach", el.ComponentType.F64),
]
"""
Scalar Mach number.
Shape: (1,)
Units: Dimensionless
Usage: Ratio of rocket's speed to speed of sound in local atmosphere.
"""

Motor = ty.Annotated[
    jax.Array,
    el.Component("rocket_motor", el.ComponentType.F64),
]
"""
Scalar rocket motor value.
Shape: (1,)
Units: N (Newtons)
Usage: Current thrust output from the rocket motor.
"""

PitchPID = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "Kp,Ki,Kd"},
    ),
]
"""
3D array for pitch PID controller gains.
Shape: (3,)
Elements: Kp (proportional), Ki (integral), Kd (derivative)
Units: Dimensionless
Usage: Gains for pitch axis PID control loop.
"""

PitchPIDState = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid_state",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "e,i,d", "priority": 18},
    ),
]
"""
3D array for pitch PID controller state.
Shape: (3,)
Elements: error, integral, derivative
Units: Dimensionless
Usage: State variables for pitch PID controller.
"""

Thrust = ty.Annotated[
    jax.Array,
    el.Component("thrust", el.ComponentType.F64, metadata={"priority": 17}),
]
"""
Scalar thrust value.
Shape: (1,)
Units: N (Newtons)
Usage: Net thrust force applied to the rocket.
"""

VRelAccel = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 20},
    ),
]
"""
3D array for relative acceleration in body frame.
Shape: (3,)
Units: m/s²
Usage: Acceleration of the rocket relative to its body axes.
"""

VRelAccelBuffer = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_buffer",
        el.ComponentType(el.PrimitiveType.F64, (lp_buffer_size, 3)),
        metadata={"priority": -1},
    ),
]
"""
Buffer of relative acceleration samples.
Shape: (lp_buffer_size, 3)
Units: m/s²
Usage: Stores recent relative acceleration samples for filtering.
"""

VRelAccelFiltered = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_filtered",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 19},
    ),
]
"""
3D array for filtered relative acceleration.
Shape: (3,)
Units: m/s²
Usage: Low-pass filtered relative acceleration in body axes, used for control and state estimation.
"""

Wind = ty.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
"""
3D array for wind vector in world frame.
Shape: (3,)
Units: m/s
Usage: Wind velocity vector applied to the rocket in the world coordinate system.
"""

AccelSetpointSmooth = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint_smooth",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 100},
    ),
]
