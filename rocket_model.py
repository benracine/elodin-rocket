import jax.numpy as jnp
from dataclasses import field
import elodin as el

from rocket_constants import lp_buffer_size
from rocket_types import (
    AngleOfAttack,
    AccelSetpoint,
    AccelSetpointSmooth,
    AeroCoefs,
    AeroForce,
    CenterOfGravity,
    DynamicPressure,
    FinControl,
    FinDeflect,
    Mach,
    Motor,
    PitchPID,
    PitchPIDState,
    Thrust,
    VRelAccel,
    VRelAccelBuffer,
    VRelAccelFiltered,
    Wind,
)

pitch_pid = [1.1, 0.8, 3.8]


@el.dataclass
class Rocket(el.Archetype):
    angle_of_attack: AngleOfAttack = field(default_factory=lambda: jnp.array([0.0]))
    aero_coefs: AeroCoefs = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    center_of_gravity: CenterOfGravity = field(default_factory=lambda: jnp.float64(0.2))
    mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
    dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(0.0))
    aero_force: AeroForce = field(default_factory=lambda: el.SpatialForce())
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    motor: Motor = field(default_factory=lambda: jnp.float64(0.0))
    fin_deflect: FinDeflect = field(default_factory=lambda: jnp.float64(0.0))
    fin_control: FinControl = field(default_factory=lambda: jnp.float64(0.0))
    v_rel_accel_buffer: VRelAccelBuffer = field(
        default_factory=lambda: jnp.zeros((lp_buffer_size, 3))
    )
    v_rel_accel: VRelAccel = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    v_rel_accel_filtered: VRelAccelFiltered = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    pitch_pid: PitchPID = field(default_factory=lambda: jnp.array(pitch_pid))
    pitch_pid_state: PitchPIDState = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    accel_setpoint: AccelSetpoint = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    accel_setpoint_smooth: AccelSetpointSmooth = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))
