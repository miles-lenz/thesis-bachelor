"""
This module manages all calculations related to MuJoCo bodies.

The main function, `calc_body_params`, returns all relevant parameters.

Other functions can be used to retrieve specific parameters as needed.
"""

from resources.mimoGrowth.constants import RATIOS_MIMO_BODIES as ratios
import numpy as np


def calc_body_positions(params_geoms: dict, age: float) -> dict:
    """
    This function will calculate the position of every body based on
    the given geom parameters.

    Arguments:
        params_geoms (dict): All relevant geom parameters.
        age (float): The age of MIMo.

    Returns:
        dict: The position of every body.
    """

    g_lb, g_cb = params_geoms["lb"], params_geoms["cb"]
    g_ub1, g_ub3 = params_geoms["ub1"], params_geoms["ub3"]
    g_head, g_hand = params_geoms["head"], params_geoms["geom:right_hand1"]
    g_u_arm, g_l_arm = params_geoms["left_uarm1"], params_geoms["left_larm"]
    g_u_leg = params_geoms["geom:right_upper_leg1"]
    g_l_leg1 = params_geoms["geom:left_lower_leg1"]
    g_l_leg2 = params_geoms["geom:left_lower_leg2"]
    g_foot = params_geoms["geom:left_foot3"]

    hip = [0, 0, 0]
    lower_body = [
        0.002,
        0,
        g_cb["size"][0] * ratios["lower_body"]
    ]
    upper_body = [
        -0.002,
        0,
        g_ub1["size"][0] * ratios["upper_body"]
    ]
    eye = ratios["eye"] * g_head["size"][0] - ((1/12000) * age - 0.001)
    head = [0, 0, (g_ub3["pos"][2] + g_ub3["size"][0]) * ratios["head"]]
    upper_arm = [
        -0.005,
        (np.sum(g_ub3["size"]) + g_u_arm["size"][0]) * ratios["upper_arm"][0],
        g_ub3["pos"][2] * ratios["upper_arm"][1]
    ]
    lower_arm = [
        0,
        0,
        (g_u_arm["size"][0] + 2 * g_u_arm["size"][1] - g_l_arm["size"][0])
        * ratios["lower_arm"]
    ]
    hand = [
        0,
        0.007,
        (g_l_arm["size"][0] + 2 * g_l_arm["size"][1]) * ratios["hand"]
    ]
    fingers = [0, 0, g_hand["size"][2] * 2]
    upper_leg = [
        0.005,
        max(
            g_u_leg["size"][0],
            (np.sum(g_lb["size"]) - g_u_leg["size"][0]) * ratios["upper_leg"]
        ),  # This avoids that the upper legs collide.
        -.007
    ]
    lower_leg = [
        0,
        0,
        -(g_u_leg["size"][0] + 2 * g_u_leg["size"][1] - g_l_leg1["size"][0])
        * ratios["lower_leg"]
    ]
    foot = [0, 0, (g_l_leg1["pos"][2] + g_l_leg2["pos"][2]) * ratios["foot"]]
    toes = [g_foot["pos"][0], 0, 0]

    positions = [
        (["hip"], hip),
        (["lower_body"], lower_body),
        (["upper_body"], upper_body),
        (["head"], head),
        (["left_eye"], eye), (["right_eye"], eye * np.array([1, -1, 1])),
        (["left_upper_arm"], upper_arm),
        (["right_upper_arm"], upper_arm * np.array([1, -1, 1])),
        (["right_lower_arm", "left_lower_arm"], lower_arm),
        (["left_hand"], hand), (["right_hand"], hand * np.array([1, -1, 1])),
        (["right_fingers", "left_fingers"], fingers),
        (["right_upper_leg"], upper_leg * np.array([1, -1, 1])),
        (["left_upper_leg"], upper_leg),
        (["right_lower_leg", "left_lower_leg"], lower_leg),
        (["right_foot", "left_foot"], foot),
        (["right_toes", "left_toes"], toes),
    ]

    return positions


def calc_body_params(params_geoms: dict, age: float) -> dict:
    """
    This function calculates all relevant body parameters based on the
    geom parameters for the given age.

    Arguments:
        params_geoms (dict): All relevant geom parameters.
        age (float): The age of MIMo.

    Returns:
        dict: All relevant body parameters. Can be accessed via body name.
    """

    body_positions = calc_body_positions(params_geoms, age)

    params = {}
    for body_names, pos in body_positions:
        for name in body_names:
            params[name] = {"pos": pos}

    return params
