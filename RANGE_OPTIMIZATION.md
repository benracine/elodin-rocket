# Rocket Range Optimization

This project includes a range optimization feature to find the optimal launch pitch angle for maximum rocket range.

## Files Added

- `range_optimizer.py` - Main optimization script that tests multiple pitch angles
- `validate_optimization.py` - Quick validation script comparing old vs optimal pitch angles

## Usage

### Run Range Optimization
```bash
cd src
python range_optimizer.py
```

This will:
- Test 17 pitch angles from 5° to 85°
- Calculate the range for each angle using physics simulation
- Find the optimal pitch angle for maximum range
- Generate a plot showing the results
- Provide recommendations for updating the simulation

### Quick Validation
```bash
cd src
python validate_optimization.py
```

This shows a quick comparison between the original and optimized pitch angles.

## Results

The optimization found that:
- **Original pitch angle**: 8.0° (achieved ~10,060 m range)
- **Optimal pitch angle**: 35.0° (achieves ~18,219 m range)
- **Improvement**: 81.1% increase in range (~8,160 m additional distance)

The main simulation (`rocket_sim.py`) has been updated to use the optimal pitch angle.

## How It Works

The range optimizer uses a physics-based trajectory simulation that includes:
- Real motor thrust curve data (AeroTech M685W)
- Actual rocket mass (8 kg total)
- Aerodynamic drag calculations
- Gravity effects
- Launch height (0.5m above ground)

The physics model integrates forces over time to compute the complete trajectory and determine where the rocket lands.

## Background

Classic ballistics theory suggests that 45° is optimal for maximum range in a vacuum, but real rockets have:
- Thrust that varies over time (motor burn profile)
- Aerodynamic drag that increases with velocity
- Changing mass as fuel is consumed

The optimization accounts for these real-world factors to find the true optimal angle of ~35°.