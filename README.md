# Ballistic Rocket Max Range Simulation

## Prerequisites

- Python 3.12
- UV package manager ([GitHub](https://github.com/astral-sh/uv))
- Elodin Sim

## Quickstart

1. Clone this repository
1. Move to this repository
1. Create and activate a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

1. Install dependencies:

   ```bash
   uv sync
   ```

1. Run the following command to launch the simulation editor:

```bash
elodin editor rocket.py
```

## Design Documentation

The design and implementation details are documented in the following RFDs:

- [RFD 0001](rfd/0001-simulation-design.md) - Ballistic Rocket Simulation Design
- [RFD 0002](rfd/0002-performance-optimization.md) - Performance Optimization Strategy

- Integration of wind effects and moving target scenarios (bonus points).
- Visualization of flight path and range results.
- Parameter sweep for optimization (e.g., launch angle, mass distribution).
- Documentation of assumptions and limitations.

### Next Steps

1. Implement the core simulation logic in `rocket.py`.
2. Add input handling for variable conditions.
3. Validate results against expected physics.
4. Expand documentation and visualization as time allows.
