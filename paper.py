"""
This module contains functions that are used exclusively to create figures
and plots for the MIMo paper.
"""

import os
import re
import argparse

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from matplotlib import colors as mcl

from show import adjust_pos
from resources.mimoGrowth.growth import adjust_mimo_to_age, delete_growth_scene
import resources.mimoEnv.utils as utils


def strength_test(body_part: str) -> None:
    """
    This function plots the performance of MIMo for the lifting-legs or
    lifting-head strength test at different ages.

    The strength test is performed by placing MIMo in a supine position and
    then fully activating the relevant actuators that to perform the movement.

    The x-axis describes the amount of simulation steps, while the y-axis
    shows the normalized hip joint angle.

    Arguments:
        body_part (str): The body part MIMo will lift.
    """

    path = "resources/mimoEnv/assets/growth.xml"
    ages = [0, 2, 4, 6, 12, 24]

    all_qpos = []
    for age in ages:

        growth_model = adjust_mimo_to_age(age, path, False)

        model = mujoco.MjModel.from_xml_path(growth_model)
        data = mujoco.MjData(model)

        delete_growth_scene(growth_model)

        model.body("growth_references").pos = [0, 0, -5]
        adjust_pos("supine", model, data)

        if body_part == "legs":
            data.ctrl[28] = -1  # right_hip_flex
            data.ctrl[36] = -1  # left_hip_flex
        elif body_part == "head":
            data.ctrl[4] = 1  # head_tilt

        # Make sure that every joint starts with an angle of zero since
        # this is not guaranteed by just bringing MIMo into supine position.
        if body_part == "legs":
            utils.set_joint_qpos(model, data, "robot:right_hip1", [0])
            utils.set_joint_qpos(model, data, "robot:left_hip1", [0])
        elif body_part == "head":
            utils.set_joint_qpos(model, data, "robot:head_tilt", [0])

        if body_part == "legs":
            joint_id = model.joint("robot:right_hip1").id
        elif body_part == "head":
            joint_id = model.joint("robot:head_tilt").id
        joint_qpos_index = model.jnt_qposadr[joint_id]

        index = 0 if body_part == "legs" else 1
        bound = model.jnt_range[joint_id][index]

        qpos_values = []
        for _ in range(500):

            qpos = data.qpos[joint_qpos_index]
            qpos_values.append(qpos / bound)

            mujoco.mj_step(model, data)

        all_qpos.append(qpos_values)

    start_color = plt.get_cmap("tab10")(0)  # tab:blue
    end_color = plt.get_cmap("tab10")(1)    # tab:orange

    cmap = mcl.LinearSegmentedColormap.from_list(
        "custom_cmap", [start_color, end_color]
    )

    color_list = [cmap(i / (len(ages) - 1)) for i in range(len(ages))]
    color_hex_list = [mcl.to_hex(c) for c in color_list]

    _, ax = plt.subplots(figsize=(24 * 0.48, 24 * 0.24))

    for i, (age, qpos) in enumerate(zip(ages, all_qpos)):
        color = color_hex_list[i]
        ax.plot(qpos, label=f"{age} month(s)", color=color, linewidth=2)

    plt.xlabel("Simulation steps")
    plt.ylabel("Normalized joint angle")

    plt.legend(loc="upper left")
    plt.show()


def create_mimo_xml(age: float) -> None:
    """
    This function creates a model and meta file for MIMo at the specified age.
    Such files can then be used to load multiple MIMos in a single scene.

    Arguments:
        age (float): The age of MIMo. Possible values are between 0 and 24.
    """

    path_scene = "resources/mimoEnv/assets/growth.xml"
    path_scene_temp = adjust_mimo_to_age(age, path_scene, False)

    dirname = os.path.dirname(path_scene_temp)
    path_model_temp = os.path.join(dirname, "mimo/MIMo_model_temp.xml")
    path_meta_temp = os.path.join(dirname, "mimo/MIMo_meta_temp.xml")

    tree_model = ET.parse(path_model_temp)
    root_model = tree_model.getroot()
    tree_meta = ET.parse(path_meta_temp)
    root_meta = tree_meta.getroot()

    # Update names in model file.
    for element in ["geom", "body", "joint", "site", "camera"]:
        for e in root_model.findall(f".//{element}"):
            e.attrib["name"] = f"{e.attrib["name"]}_{age}"

    # Remove elements in meta file.
    # Instead, these elements are provided within the scene XML file.
    for element in ["default", "asset"]:
        root_meta.remove(root_meta.find(element))

    elements = [
        "fixed", "joint", "motor",
        "torque", "gyro", "accelerometer",
        "pair", "exclude"
    ]
    attributes = [
        "name", "joint", "joint1", "joint2",
        "geom1", "geom2", "body1", "body2", "site"
    ]

    # Update names in meta file.
    for element in elements:
        for e in root_meta.findall(f".//{element}"):
            for att in attributes:
                if att in e.attrib:
                    e.attrib[att] = f"{e.attrib[att]}_{age}"

    tree_model.write(os.path.join(dirname, f"mimo_ages/MIMo_model_{age}.xml"))
    tree_meta.write(os.path.join(dirname, f"mimo_ages/MIMo_meta_{age}.xml"))

    delete_growth_scene(path_scene_temp)


def mimo_stages(
        age_mimo: float = None,
        passive: bool = None,
        save: bool = False) -> None:
    """
    This method is used to create an image of multiple MIMos at various ages
    with different poses.

    Note that this script assumes that the only joints in the scene belong
    to MIMo. Otherwise, the script will break.

    Arguments:
        age_mimo (float): The age of MIMo. It is important to the associated
            files are within the 'mimo_ages' folder.
        passive (bool): If the MuJoCo viewer should be passive.
        save (bool): If the image should be saved. If save=True the mujoco
            viewer will not be launched.
    """

    path = "resources/mimoEnv/assets/mimo_stages.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    ages_order = []
    for body in root.findall(".//body"):
        if re.search(r'\d+', body.attrib["name"]):
            ages_order.append(body.attrib["name"].split("_")[-1])

    if age_mimo is not None:

        for body in root.findall(".//body"):
            if not re.search(r'\d+', body.attrib["name"]):
                continue
            if str(age_mimo) not in body.attrib["name"]:
                root.find("worldbody").remove(body)
            # else:
            #     pos_z = body.attrib["pos"].split(" ")[-1]
            #     body.attrib["pos"] = f"0 0 {pos_z}"

        for include in root.findall(".//include"):
            if str(age_mimo) not in include.attrib["file"]:
                root.remove(include)

        path = path.replace(".xml", "_temp.xml")
        tree.write(path)

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    if age_mimo is not None:
        os.remove(path)

    tree = ET.parse("resources/mimoEnv/assets/mimo_ages/qpos_stages.xml")
    root = tree.getroot()

    qpos_by_age = {}
    for age_group in root.findall(".//age_group"):
        age = age_group.attrib["name"]
        qpos_by_age[age] = age_group.find("key").attrib["qpos"].split(" ")

    # Store indices of the free joints and ignore them in the next step.
    # This allows to modify position/rotation of MIMo via the scene.
    free_joint_indices = np.array([
        np.array([0, 1, 3, 4, 5, 6]) + 54 * i
        for i in range(0, 5)
    ]).flatten()

    if age_mimo is None:

        qpos = []
        for age in ages_order:
            qpos += qpos_by_age[age]

        for i in range(model.nq):
            if i not in free_joint_indices:
                data.qpos[i] = qpos[i]

    else:

        for i in range(model.nq):
            if i not in free_joint_indices:
                data.qpos[i] = float(qpos_by_age[str(age_mimo)][i])

    mujoco.mj_forward(model, data)

    if save:

        width, height = 1920, 1080
        renderer = mujoco.Renderer(model, width=width, height=height)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="main")
        img = renderer.render()

        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output.png", image_bgr)

        return

    launch = mujoco.viewer.launch_passive if passive else mujoco.viewer.launch
    with launch(model, data) as viewer:
        while viewer.is_running():
            pass


def mimo_detail(passive: bool = False) -> None:
    """
    This function shows MIMo holding a ball at his youngest and oldest age.

    Arguments:
        passive (bool): If the MuJoCo viewer should be passive.
    """

    path = "resources/mimoEnv/assets/mimo_detail.xml"

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    tree = ET.parse("resources/mimoEnv/assets/mimo_ages/qpos_detail.xml")
    root = tree.getroot()

    qpos = root.attrib["qpos"].split(" ")
    for i in range(model.nq):
        data.qpos[i] = qpos[i]

    launch = mujoco.viewer.launch_passive if passive else mujoco.viewer.launch
    with launch(model, data) as viewer:
        while viewer.is_running():
            pass


if __name__ == "__main__":

    func_map = {
        "strength_test": strength_test,
        "create_mimo_xml": create_mimo_xml,
        "mimo_stages": mimo_stages,
        "mimo_detail": mimo_detail
    }

    parser = argparse.ArgumentParser(
        description="Run functions from the terminal."
    )
    parser.add_argument(
        "function",
        choices=func_map.keys(),
        help="The function to call."
    )
    parser.add_argument(
        "kwargs",
        nargs=argparse.REMAINDER,
        help="Additional keyword arguments."
    )

    kwargs = {}
    for param in parser.parse_args().kwargs:
        key, value = param.split("=")
        try:
            kwargs[key] = eval(value)
        except (NameError, SyntaxError):
            kwargs[key] = value

    func_map[parser.parse_args().function](**kwargs)
