# RFD: Pitch Sweep with Polars

## Problem

We need to vary th pitch angle that gives the max range. The current sim only runs once.
At the moment, I'm hitting an issue that prevents me from running multiple runs in sequence.

## Plan

* Define a list of pitch angles.
* For each pitch, run the sim and record every timestep.
* Store results in a single Polars DataFrame with columns:

  * `pitch_deg`
  * `t` - time step
  * `x`
  * `z`

## Analysis

* For each pitch, find the last timestep where `z >= 0`.
* Take that `x` as the range.
* Select the pitch and range combination with the largest range.

## Next Steps

Implement a simple loop:

* Collect results into one Polars DataFrame.
* E.g. find the greatest range associated with a positive Z.
* Refine the analysis to include an interpolation process when the position flips from positive Z to negative Z between a time step in order to refine the accuract a bit.

Also, explore Monte Carlo simulations. I had assumed that those only allowed for random perturbations in inputs and required connecting to AWS, but I haven't yet had time to explore that.
