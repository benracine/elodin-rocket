# Ballistic Rocket Range Simulation

## Prerequisites

- Python 3.12
- UV package manager ([GitHub](https://github.com/astral-sh/uv))
- Elodin Sim

## Quickstart

1. Clone [this repository](https://github.com/benracine/elodin-rocket) to your machine.
1. Move to the newly cloned local repository on the command line.
1. Create and activate a virtual environment like so:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

1. Install the necessary dependencies (to enable further development) like so:

   ```bash
   uv sync
   ```

1. Run the suite of simulations using:

```bash
uv run python src/main.py run
```
