from rocket_sim import create_world, create_system, SIM_TIME_STEP
import numpy as np
# import polars as pl
import elodin as el
# import matplotlib.pyplot as plt

def main():
    # create jax array range of pitch angles from 0 to 90 degrees by 5 degree increments
    # pitch_angles = jnp.arange(0, 91, 5)
    pitch_angles = np.array([15, 30])
    # Run the simulation over the range of pitch angles
    for pitch in pitch_angles:
        print(f"\n--- Running simulation for pitch angle: {pitch} degrees ---")
        w = create_world(pitch)
        sys = create_system()
        w.run(sys, sim_time_step=SIM_TIME_STEP, max_ticks=1200)

        el.Exec(w).history("Rocket.world_pos")

        """
        df = w.history("Rocket.world_pos")
        df = df.with_columns(
            pl.col("Rocket.world_pos").arr.get(4).alias("Rocket.x"),
            pl.col("Rocket.world_pos").arr.get(5).alias("Rocket.y"),
            pl.col("Rocket.world_pos").arr.get(6).alias("Rocket.z"),
        )
        print(df)
        """

    """
    distance = np.linalg.norm(df.select(["Rocket.x", "Rocket.y", "Rocket.z"]).to_numpy(), axis=1)
    df = df.with_columns(pl.Series(distance).alias("distance"))
    ticks = np.arange(df.shape[0])
    fig, ax = plt.subplots()
    ax.plot(ticks, df["distance"])
    plt.show()
    """

if __name__ == "__main__":
    print("Starting simulation...")
    main()