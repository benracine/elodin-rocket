from rocket_sim import create_world, create_system, SIM_TIME_STEP
import numpy as np
import polars as pl
import elodin as el
# import matplotlib.pyplot as plt


def main():
    # Explore pitch angle array
    # jnp.arange(0, 91, 5)
    pitch_angles = np.array([15, 30])

    # Use monte carlo approach to find optimal launch angle for max range

    # Run the simulation over the range of pitch angles
    for pitch in pitch_angles:
        print(f"\n--- Running simulation for pitch angle: {pitch} degrees ---")
        w = create_world(pitch)
        sys = create_system()
        w.run(sys, sim_time_step=SIM_TIME_STEP, max_ticks=1200, optimize=True)

        # Investigate the historical data
        # Unable to get here for some reason I wasn't yet able to debug
        el.Exec(w).history("Rocket.world_pos")

        df = w.history("Rocket.world_pos")
        df = df.with_columns(
            pl.col("Rocket.world_pos").arr.get(4).alias("Rocket.x"),
            pl.col("Rocket.world_pos").arr.get(5).alias("Rocket.y"),
            pl.col("Rocket.world_pos").arr.get(6).alias("Rocket.z"),
        )
        print(df)

    # After running all simulations, find the max range where z is still positive


if __name__ == "__main__":
    print("Starting simulation...")
    main()
