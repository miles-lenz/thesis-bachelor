""" This module store utility and helper functions. """

from resources.mimoGrowth.constants import AGE_GROUPS, RATIOS_MIMO_GEOMS, \
    CHILDREN_MEASUREMENTS
import re
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import xml.etree.ElementTree as ET


def growth_function(x, a, b, c) -> float:
    """
    This function represents the standard form of the growth functions.

    By default, this is a logarithmic function. If you want to explore
    different types of approximations, simply modify the return statement
    to use other mathematical expressions (e.g., a quadratic function).

    Example: Use `a * x ** 2 + b * x + c` if you want to try a
    quadratic function.

    Notice that the bounds in `approximate_growth_functions` should be
    changed accordingly.

    Arguments:
        x (float): The input value for which the function is evaluated.
            This represents the age of MIMo and will be between 0 and 24.
        a, b, c (float): Parameters, that will modify the function.

    Returns:
        float: The result of the function evaluation at the given `x`.
    """

    return a * np.log(x + b) + c


def load_measurements() -> dict:
    """
    This function loads and returns relevant data from the measurements folder.
    A single measurement list matches the length of the age list in the
    constant.py file.

    The original measurements can be found on the following website:
    https://math.nist.gov/~SRessler/anthrokids/

    Returns:
        dict: Every key-value pair describes one body part and its growth.
    """

    path_script = os.path.dirname(os.path.realpath(__file__))
    path_meas = os.path.join(path_script, "measurements/")

    measurements = {}
    for file_name in next(os.walk(path_meas))[2]:

        df = pd.read_csv(path_meas + file_name)
        children_meas = CHILDREN_MEASUREMENTS[file_name[:-4]]

        measurements[file_name[:-4]] = {
            "mean": df.MEAN.to_list() + [children_meas[0]],
            "std": df["S.D."].tolist() + [children_meas[1]],
        }

    return measurements


def approximate_growth_functions(measurements: dict) -> dict:
    """
    This function approximates a growth functions for each body part based
    on the measurements.

    Arguments:
        measurements (dict): The measurements for all body parts.

    Returns:
        dict: A growth function for each body part.
    """

    config = {
        "maxfev": 10000,
        # Use bounds for the log function to avoid the issue of log(0).
        "bounds": [(-np.inf, 0.1, -np.inf), (np.inf, np.inf, np.inf)]
    }

    functions = {}
    for body_part, meas in measurements.items():

        x, y = AGE_GROUPS, meas["mean"]
        params = curve_fit(growth_function, x, y, **config)[0]

        functions[body_part] = params

    return functions


def estimate_sizes(functions: dict, age: float) -> dict:
    """
    This function uses the approximated functions and the given age
    to estimate sizes for each body part.

    Arguments:
        functions (dict): The growth functions for all body parts.
        age (float): The age of MIMo.

    Returns:
        dict: The predicted size for every body part at the given age.
    """

    sizes = {}
    for body_part, params in functions.items():
        sizes[body_part] = growth_function(age, *params)

    return sizes


def format_sizes(sizes: dict) -> dict:
    """
    This function will format the estimated sizes.
    Specifically, this means:
    - Converting units to MuJoCo standards
    - Group measurements so they can be associated with a geom
    - Applying ratios

    This list describes the high-level body parts and the
    corresponding measurements:
    - head      : Head Circumference
    - upper_arm : [Upper Arm Circumference, Shoulder Elbow Length]
    - lower_arm : [Forearm Circumference, Elbow Hand Length - Hand Length]
    - hand      : [Hand Length, Hand Breadth, Maximum Fist Breadth]
    - torso     : Hip Breadth
    - upper_leg : [Mid Thigh Circumference, Rump Knee Length]
    - lower_leg : [Calf Circumference, Ankle Circumference, Knee Sole Length]
    - foot      : [Foot Length, Foot Breadth]

    Arguments:
        sizes (dict): The estimated sizes for all body parts.

    Returns:
        dict: The formatted sizes for all body parts.
    """

    # Use meter as unit and convert circumference to radius or
    # split lengths in half. MuJoCo expects these units.
    for body_part, meas in sizes.items():
        sizes[body_part] = np.array(meas) / 100
        sizes[body_part] /= 2 * np.pi if "circum" in body_part else 2

    # Group the measurements. This will make later calculations easier.
    # Notice that for some body parts we need to subtract the radius from the
    # length since MuJoCo expects the half-length only of the cylinder part.
    sizes = {
        "head": [sizes["head_circumference"]],
        "upper_arm": [
            sizes["upper_arm_circumference"],
            sizes["shoulder_elbow_length"] - sizes["upper_arm_circumference"]
        ],
        "lower_arm": [
            sizes["forearm_circumference"],
            (
                sizes["elbow_hand_length"] -
                sizes["hand_length"] -
                sizes["forearm_circumference"]
            )
        ],
        "hand": [
            sizes["hand_length"],
            sizes["hand_breadth"],
            sizes["maximum_fist_breadth"]
        ],
        # For the torso we need to duplicate the size by five
        # since the whole torso is made up of five capsules.
        # Each capsule will be tweaked a little by the ratio later.
        "torso": np.repeat(sizes["hip_breadth"], 5),
        "upper_leg": [
            sizes["mid_thigh_circumference"],
            sizes["rump_knee_length"] - sizes["mid_thigh_circumference"]
        ],
        "lower_leg": [
            sizes["calf_circumference"],
            sizes["ankle_circumference"],
            (
                sizes["knee_sole_length"] -
                sizes["calf_circumference"] / 2 -
                sizes["ankle_circumference"] / 2
            )
        ],
        "foot": [sizes["foot_length"], sizes["foot_breadth"]]
    }

    for body_part in sizes.keys():
        sizes[body_part] *= np.array(RATIOS_MIMO_GEOMS[body_part])

    return sizes


def calc_volume(size: list, geom_type: str) -> float:
    """
    This function returns the volume based on the size and type of a geom.

    Arguments:
        size (list): The size of the geom.
        geom_type (str): The type of the geom. This needs to be one of the
        following: 'sphere', 'capsule' or 'box'

    Returns:
        float: The volume of the geom.

    Raises:
        ValueError: If the geom type is invalid.
    """

    if geom_type == "sphere":
        vol = (4 / 3) * np.pi * size[0] ** 3

    elif geom_type == "capsule":
        vol = (4 / 3) * np.pi * size[0] ** 3
        vol += np.pi * size[0] ** 2 * size[1] * 2

    elif geom_type == "box":
        vol = np.prod(size) * 8

    elif geom_type == "cylinder":
        vol = np.pi * size[0] ** 2 * size[1] * 2

    else:
        raise ValueError(f"Unknown geom type '{geom_type}'.")

    return vol


def store_base_values(path_scene: str) -> None:
    """
    This function stores relevant values of the original MIMo model before
    the age is changed.

    Arguments:
        path_scene (str): The path to the MuJoCo scene.

    Returns:
        dict: All relevant values of MIMo.
    """

    base_values = {"geom": {}, "motor": {}}

    tree_scene = ET.parse(path_scene)

    includes = {}
    for include in tree_scene.getroot().findall(".//include"):
        key = "model" if "model" in include.attrib["file"] else "meta"
        includes[key] = include

    path_dir = os.path.dirname(path_scene)
    path_model = os.path.join(path_dir, includes["model"].attrib["file"])
    path_meta = os.path.join(path_dir, includes["meta"].attrib["file"])

    tree_model = ET.parse(path_model)
    tree_meta = ET.parse(path_meta)

    for geom in tree_model.getroot().findall(".//geom"):

        type_ = geom.attrib["type"]

        size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
        size = np.array(size.split(" "), dtype=float)

        vol = calc_volume(size, type_)
        density = float(geom.attrib["mass"]) / vol

        if type_ in ["sphere", "capsule"]:
            csa = np.pi * size[0] ** 2
        elif type_ == "box":
            csa = size[0] * size[1] * 4

        base_values["geom"][geom.attrib["name"]] = {
            "type": type_,
            "size": size,
            "vol": vol,
            "csa": csa,
            "density": density,
        }

    for motor in tree_meta.getroot().find("actuator").findall("motor"):

        base_values["motor"][motor.attrib["name"]] = {
            "gear": float(motor.attrib["gear"])
        }

    return base_values
