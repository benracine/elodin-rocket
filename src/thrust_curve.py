"""
thrust_curve.py

Parse AeroTech_M685W.eng and convert to thrust_curve dict for simulation use.
"""

import os


def parse_thrust_curve(filepath):
    """
    Parse a .eng file and return a thrust curve dictionary.

    Args:
        filepath (str): Path to the .eng file.

    Returns:
        dict: Dictionary with 'time' and 'thrust' lists.
    """
    time = []
    thrust = []
    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(";") or line[0].isalpha():
                continue
            parts = line.split()
            if len(parts) == 2:
                t, f_val = map(float, parts)
                time.append(t)
                thrust.append(f_val)
    return {"time": time, "thrust": thrust}


ENG_PATH = os.path.join(os.path.dirname(__file__), "../assets/AeroTech_M685W.eng")
thrust_curve = parse_thrust_curve(ENG_PATH)
