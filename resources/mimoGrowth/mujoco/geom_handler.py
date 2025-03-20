"""
This module manages all calculations related to MuJoCo geoms.

The main function, `calc_geom_params`, returns all relevant parameters.

Other functions can be used to retrieve specific parameters as needed.
"""

from resources.mimoGrowth.constants import RATIOS_DERIVED as ratios
from resources.mimoGrowth.constants import MAPPING_GEOM
from resources.mimoGrowth.utils import calc_volume
import numpy as np


# Use a small constant to subtract from some geom vectors
# so that the individual parts won't have a visual overlap.
# This value is from the original MIMo model.
EPSILON = 0.0001


def calc_extras(approx_sizes: dict) -> dict:
    """
    This function will calculate some needed values that can not be obtained
    directly from the website.

    Hand:
    - Compute height based on geometric mean of length and breadth
    - Split length into palm and finger length based on ratio
        from original MIMo model

    Foot:
    - Compute height based on geometric mean of length and breadth
    - Split length into foot and toes based on ratio from original MIMo model

    Lower Leg:
    - Split the length since the original model uses two capsules for the
        lower leg. Use ratio from original model to do this.

    Arguments:
        approx_sizes (dict): The estimated sizes for all body
            parts at the given age.

    Returns:
        dict: Additional values.
    """

    extras = {}

    # ===== HAND =====

    l_hand, hand_breadth, _ = approx_sizes["hand"]

    h_hand = np.sqrt(l_hand * hand_breadth) * ratios["hand"][0]
    l_hand -= (h_hand + EPSILON * 2) / 2

    len_hand = l_hand * ratios["hand"][1]
    len_fingers = l_hand * (1 - ratios["hand"][1])

    extras["hand"] = [h_hand, len_hand, len_fingers]

    # ===== FOOT =====

    h_foot = np.sqrt(np.prod(approx_sizes["foot"])) * ratios["foot"][0]

    len_foot = approx_sizes["foot"][0] * ratios["foot"][1]
    len_toes = approx_sizes["foot"][0] * ratios["foot"][2]

    extras["foot"] = [h_foot, len_foot, len_toes]

    # ===== LOWER LEG =====

    len_lower_leg = approx_sizes["lower_leg"][2]

    # Subtract the foot height from the length since the measurements
    # from the website include the foot but in the MIMo model we handle
    # lower leg and foot separately.
    len_lower_leg -= h_foot

    len_lower_leg1 = len_lower_leg * ratios["lower_leg"][0]
    len_lower_leg2 = len_lower_leg * (1 - ratios["lower_leg"][0])

    extras["lower_leg"] = [len_lower_leg1, len_lower_leg2]

    return extras


def calc_geom_sizes(approx_sizes: dict, extras: dict) -> dict:
    """
    This function will calculate the size of all geoms based on the
    estimated sizes and some extra values.

    Arguments:
        approx_sizes (dict): The estimated sizes for all body
            parts at the given age.
        extras (dict): Additional values that could not be obtained
            from the website.

    Returns:
        dict: The size of every geom.
    """

    h_hand, len_hand, len_fingers = extras["hand"]
    h_foot, len_foot, len_toes = extras["foot"]
    len_lower_leg1, len_lower_leg2 = extras["lower_leg"]

    geom_sizes = {
        "head": [
            approx_sizes["head"],  # head
            approx_sizes["head"] * ratios["eye"]  # eye
        ],
        "upper_arm": [approx_sizes["upper_arm"]],
        "lower_arm": [approx_sizes["lower_arm"]],
        "hand": [
            # hand1
            [approx_sizes["hand"][2], h_hand, len_hand],
            # hand2
            [h_hand + EPSILON * 2, approx_sizes["hand"][2] - EPSILON * 3, 0],
            # fingers1
            [approx_sizes["hand"][1], h_hand, len_fingers],
            # fingers2
            [h_hand + EPSILON * 2, approx_sizes["hand"][1] + EPSILON * 2, 0],
        ],
        "upper_leg": [approx_sizes["upper_leg"]],
        "lower_leg": [
            [approx_sizes["lower_leg"][0], len_lower_leg1],  # lower_leg1
            [approx_sizes["lower_leg"][1], len_lower_leg2]  # lower_leg2
        ],
        "foot": [
            # foot1
            [approx_sizes["foot"][1] - EPSILON, h_foot - EPSILON],
            # foot2
            [len_foot, approx_sizes["foot"][1], h_foot],
            # foot3
            [h_foot - EPSILON, approx_sizes["foot"][1] - EPSILON * 2],
            # toes1
            [len_toes, approx_sizes["foot"][1] - EPSILON, h_foot - EPSILON],
            # toes2
            [h_foot, approx_sizes["foot"][1]],
        ]
    }

    geom_sizes["torso"] = []
    for i in range(5):
        size, ratio = approx_sizes["torso"][i], ratios["torso_size"][i]
        vec = [size * ratio, size * (1 - ratio)]
        geom_sizes["torso"].append(vec)

    # Add a padding so there are no conflicts with MuJoCo.
    for name, vectors in geom_sizes.items():
        for i, vec in enumerate(vectors):
            geom_sizes[name][i] = np.pad(vec, (0, 3 - len(vec)))

    return geom_sizes


def calc_geom_positions(
        approx_sizes: dict, geom_sizes: dict, extras: dict) -> dict:
    """
    This function will calculate the position of every geom based on the
    estimated sizes and some extra values.

    Arguments:
        approx_sizes (dict): The estimated sizes for all body
            parts at the given age.
        geom_sizes (dict): The size of every geom.
        extras (dict): Additional values that could not be obtained
            from the website.

    Returns:
        dict: The position of every geom.
    """

    h_hand, len_hand, len_fingers = extras["hand"]
    _, len_foot, len_toes = extras["foot"]
    len_lower_leg1, len_lower_leg2 = extras["lower_leg"]
    torso_size = geom_sizes["torso"]

    positions = {
        "head": [
            [0.01, 0, approx_sizes["head"][0]],  # head
            [0, 0, 0]  # eye
        ],
        "upper_arm": [[0, 0, approx_sizes["upper_arm"][1]]],
        "lower_arm": [[0, 0, approx_sizes["lower_arm"][1]]],
        "hand": [
            [h_hand / 2, 0, len_hand],  # hand1
            [h_hand / 2, 0, len_hand * 2],  # hand2
            [0, 0, len_fingers],  # fingers1
            [0, 0, len_fingers * 2]  # fingers2
        ],
        "torso": [
            [-0.002, 0, torso_size[0][0] * ratios["torso_pos"][0]],  # lb
            [0.005, 0, torso_size[1][0] * ratios["torso_pos"][1]],  # cb
            [0.007, 0, torso_size[2][0] * ratios["torso_pos"][2]],  # ub1
            [0.004, 0, torso_size[3][0] * ratios["torso_pos"][3]],  # ub2
            [0, 0, torso_size[4][0] * ratios["torso_pos"][4]],  # ub3
        ],
        "upper_leg": [[
            0,
            0,
            -approx_sizes["upper_leg"][1] * ratios["upper_leg"]
        ]],
        "lower_leg": [
            # lower_leg1
            [0, 0, -len_lower_leg1],
            # lower_leg2
            [
                0,
                0,
                (
                    -len_lower_leg1 * 2 - len_lower_leg2 -
                    approx_sizes["lower_leg"][1]
                ) * ratios["lower_leg"][1]
            ]
        ],
        "foot": [
            # foot1
            [-len_foot * ratios["foot"][3], 0, 0],
            # foot2
            [len_foot - len_foot * ratios["foot"][3], 0, 0],
            # foot3
            [len_foot + (len_foot - len_foot * ratios["foot"][3]), 0, 0],
            # toes1
            [len_toes, 0, 0],
            # toes2
            [len_toes * 2, 0, 0]
        ]
    }

    return positions


def calc_geom_masses(params: dict, base_values: dict):
    """
    This function will calculate the mass of every geom based on the size
    and type of geom. The mass will be inserted into the 'params' dict.

    Arguments:
        params (dict): Size and position of every geom.
        base_values (dict): Relevant values from the original model.
    """

    for geom_name, attributes in params.items():

        geom_type = base_values[geom_name]["type"]
        size = attributes["size"]
        vol = calc_volume(size, geom_type)

        attributes["mass"] = vol * base_values[geom_name]["density"]


def calc_geom_params(approx_sizes: dict, base_values: dict) -> dict:
    """
    This function calculates all relevant geom parameters based on the
    estimated sizes for the given age and base values of the original MIMo.

    Arguments:
        approx_sizes (dict): The estimated sizes for all body
            parts at the given age.
        base_values (dict): Relevant values from the original MIMo.

    Returns:
        dict: All relevant geom parameters. Can be accessed via geom name.
    """

    extras = calc_extras(approx_sizes)

    geom_sizes = calc_geom_sizes(approx_sizes, extras)
    geom_positions = calc_geom_positions(approx_sizes, geom_sizes, extras)

    params = {}
    for body_part, geom_names in MAPPING_GEOM.items():
        for i, name in enumerate(geom_names):
            keys = name if isinstance(name, tuple) else [name]
            for key in keys:
                params[key] = {
                    "size": geom_sizes[body_part][i],
                    "pos": geom_positions[body_part][i]
                }

    calc_geom_masses(params, base_values["geom"])

    return params
