"""
This module contains functions that are used exclusively to create figures
and plots for the MIMo paper.
"""

from show import adjust_pos
from resources.mimoGrowth.growth import adjust_mimo_to_age, delete_growth_scene
import resources.mimoEnv.utils as utils
import argparse
import os
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt


def strength_test() -> None:
    """
    This function plots the performance of MIMo for the lifting-legs strength
    test at different ages.

    The strength test is performed by placing MIMo in a supine position and
    then fully activating the relevant hip actuators that lift his legs.

    The x-axis describes the amount of simulation steps, while the y-axis
    shows the normalized hip joint angle.
    """

    path = "resources/mimoEnv/assets/growth.xml"
    ages = [0, 2, 4, 6, 12, 18, 24]

    all_qpos = []
    for age in ages:

        growth_model = adjust_mimo_to_age(age, path, False)

        model = mujoco.MjModel.from_xml_path(growth_model)
        data = mujoco.MjData(model)

        delete_growth_scene(growth_model)

        model.body("growth_references").pos = [0, 0, -5]
        adjust_pos("supine", model, data)

        data.ctrl[28] = -1  # right_hip_flex
        data.ctrl[36] = -1  # left_hip_flex

        # Make sure that every joint starts with an angle of zero since
        # this is not guaranteed by just bringing MIMo into supine position.
        utils.set_joint_qpos(model, data, "robot:right_hip1", [0])
        utils.set_joint_qpos(model, data, "robot:left_hip1", [0])

        joint_id = model.joint("robot:right_hip1").id
        joint_qpos_index = model.jnt_qposadr[joint_id]

        bound = model.jnt_range[joint_id][0]

        qpos_values = []
        for _ in range(500):

            qpos = data.qpos[joint_qpos_index]
            qpos_values.append(qpos / bound)

            mujoco.mj_step(model, data)

        all_qpos.append(qpos_values)

    for age, qpos in zip(ages, all_qpos):
        plt.plot(qpos, label=f"{age} month(s)")

    plt.xlabel("Simulation Steps")
    plt.ylabel("Normalized Joint Angle (Radians)")

    plt.legend()
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


if __name__ == "__main__":

    func_map = {
        "strength_test": strength_test,
        "create_mimo_xml": create_mimo_xml
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
