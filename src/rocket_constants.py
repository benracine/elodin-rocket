"""
rocket_constants.py

Shared constants for rocket simulation modules.
"""

import jax.numpy as jnp

thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])
a_ref = 24.89130 / 100**2
l_ref = 5.43400 / 100
xmc = 0.25

SIM_TIME_STEP = 1.0 / 120.0
lp_sample_freq = round(1.0 / SIM_TIME_STEP)
lp_buffer_size = lp_sample_freq * 4
lp_cutoff_freq = 1
