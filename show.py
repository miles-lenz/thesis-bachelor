""" This module contains various functions for displaying MIMo. """

from resources.mimoGrowth.growth import adjust_mimo_to_age, \
    delete_growth_scene, calc_growth_params
import resources.mimoEnv.utils as mimo_utils
import resources.mimoGrowth.utils as utils
from resources.mimoGrowth.mujoco.geom_handler import calc_geom_params
from resources.mimoGrowth.mujoco.body_handler import calc_body_params
import time
import argparse
import os
import copy
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET


SCENE_PATH = "resources/mimoEnv/assets/growth.xml"


def adjust_pos(pos: str, model: MjModel, data: MjData) -> None:
    """
    This function will bring MIMo into the specified starting position.

    Possible positions are:
    - stand
    - prone, prone_on_arms, supine
    - sit, sit_forward, sit_backward, sit_left, sit_right

    Arguments:
        pos (str): The position MIMo should adapt.
        model (MjModel): The MuJoCo model.
        data (MjData): The MuJoCo data.
    """

    if pos == "stand":

        height = sum([
            -model.body("left_upper_leg").pos[2],
            -model.body("left_lower_leg").pos[2],
            -model.body("left_foot").pos[2],
            model.geom("geom:left_foot2").size[2]
        ])

        qpos = [0, 0, height, 0, 0, 0, 0]
        mimo_utils.set_joint_qpos(model, data, "mimo_location", qpos)

    elif pos in ["prone", "prone_on_arms", "supine"]:

        qpos = [0, 0, 0.2, 0, -0.7071068, 0, 0.7071068]
        if pos == "supine":
            qpos[4] *= -1

        mimo_utils.set_joint_qpos(model, data, "mimo_location", qpos)

        for _ in range(100):
            mujoco.mj_step(model, data)

        if pos == "prone_on_arms":

            joint_values = [
                ("robot:hip_bend1", [-0.25]),
                ("robot:hip_bend2", [-0.25]),
                ("robot:head_tilt", [-0.9]),
                ("robot:right_shoulder_horizontal", [1.2]),
                ("robot:left_shoulder_horizontal", [1.2]),
                ("robot:right_shoulder_ad_ab", [1.2]),
                ("robot:left_shoulder_ad_ab", [1.2]),
                ("robot:right_shoulder_rotation", [-1.3]),
                ("robot:left_shoulder_rotation", [-1.3]),
                ("robot:right_elbow", [-1.2]),
                ("robot:left_elbow", [-1.2]),
            ]

            for name, val in joint_values:
                mimo_utils.set_joint_qpos(model, data, name, val)

            for _ in range(30):
                mujoco.mj_step(model, data)

    elif "sit" in pos:

        height = data.geom("lb").xpos[2] + model.geom("lb").size[0]
        height -= model.geom("lb").pos[2]
        qpos = [0, 0, height, 0, 0, 0, 0]

        mimo_utils.set_joint_qpos(model, data, "mimo_location", qpos)
        mimo_utils.set_joint_qpos(model, data, "robot:right_hip1", [-1.58])
        mimo_utils.set_joint_qpos(model, data, "robot:left_hip1", [-1.58])

        for _ in range(5):
            mujoco.mj_step(model, data)

        if "_" in pos:

            factor, direction = 2, pos.split("_")[1]

            joint_values = {
                "forward": [
                    ("robot:hip_bend1", [0.1 * factor]),
                    ("robot:hip_bend2", [0.1 * factor])
                ],
                "backward": [
                    ("robot:hip_bend1", [-0.1 * factor]),
                    ("robot:hip_bend2", [-0.1 * factor])
                ],
                "left": [
                    ("robot:hip_lean1", [-0.1 * factor]),
                    ("robot:hip_lean2", [-0.1 * factor]),
                    ("robot:left_fingers", [-2.5]),
                ],
                "right": [
                    ("robot:hip_lean1", [0.1 * factor]),
                    ("robot:hip_lean2", [0.1 * factor]),
                    ("robot:right_fingers", [-2.5]),
                ]
            }

            for name, val in joint_values[direction]:
                mimo_utils.set_joint_qpos(model, data, name, val)

    else:
        raise ValueError(f"Unknown position '{pos}'.")


def update_mimo(model: MjModel, data: MjData, growth_params: dict) -> None:
    """
    This function will update the appearance of MIMo based on
    the provided growth parameters.

    Note that only sizes and positions are adjusted. This will
    have no effect on the mass or strength of MIMo.

    Arguments:
        model (MjModel): The MuJoCo model.
        data (MjData): The MuJoCo data.
        growth_params (dict): Growth parameters for a specific age.
    """

    for geom_name, params in growth_params["geom"].items():
        model.geom_size[model.geom(geom_name).id] = params["size"]
        model.geom(geom_name).pos = params["pos"]

    for body_name, params in growth_params["body"].items():
        model.body(body_name).pos = params["pos"]

    mujoco.mj_forward(model, data)


def growth() -> None:
    """
    This function shows the growth of MIMo within an
    interactive MuJoCo viewer.

    Hit space bar to toggle the growth and click 'strg' to
    reset the simulation.
    """

    state = {"paused": True, "reset": False}

    def key_callback(keycode):
        if keycode == 32:  # space
            state["paused"] = not state["paused"]
        elif keycode == 341:  # strg
            state["reset"] = True

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    measurements = utils.load_measurements()
    functions = utils.approximate_growth_functions(measurements)

    def calc_params(age_months, path_scene):

        approx_sizes = utils.estimate_sizes(functions, age_months)
        approx_sizes = utils.format_sizes(approx_sizes)

        base_values = utils.store_base_values(path_scene)

        params_geoms = calc_geom_params(approx_sizes, base_values)
        params_bodies = calc_body_params(params_geoms, age_months)

        return {"geom": params_geoms, "body": params_bodies}

    age_months = 0
    growth_params = calc_params(age_months, SCENE_PATH)
    update_mimo(model, data, growth_params)
    adjust_pos("stand", model, data)

    AGES_ON_CUBE = [0, 3, 6, 9, 12, 15, 18, 21, 24]

    mat_age_cube = {}
    for age in AGES_ON_CUBE:
        mat_age_cube[age] = model.material(f"age_{age}").id

    args = {"model": model, "data": data, "key_callback": key_callback}
    with mujoco.viewer.launch_passive(**args) as viewer:
        while viewer.is_running():

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(0.02)

            if state["reset"]:
                age_months = 0
                growth_params = calc_params(age_months, SCENE_PATH)
                update_mimo(model, data, growth_params)
                adjust_pos("stand", model, data)
                state["reset"], state["paused"] = False, True
                model.geom("ref_age").matid = mat_age_cube[age_months]
                continue

            if state["paused"] or age_months >= 24:
                continue

            age_months = np.round(age_months + 0.05, 2)
            growth_params = calc_params(age_months, SCENE_PATH)
            update_mimo(model, data, growth_params)
            adjust_pos("stand", model, data)

            if age_months in mat_age_cube.keys():
                model.geom("ref_age").matid = mat_age_cube[age_months]


def multiple_mimos() -> None:
    """
    This function will display multiple MIMos at different ages.

    Note that this function is only for aesthetic purposes. It should
    not be used for reinforcement learning since it might affect the
    original behavior of MIMo.
    """

    AGES = [0, 12, 24, 18, 6]
    ASSETS_PATH = "resources/mimoEnv/assets/"
    PATH_SCENE_TEMP = ASSETS_PATH + "multiple_mimos.xml"

    temp_files = [PATH_SCENE_TEMP]

    scene = ET.parse(SCENE_PATH).getroot()
    sc_worldbody = scene.find("worldbody")
    sc_body = sc_worldbody.find("body[@name='mimo_location']")
    sc_include_meta = scene.find("include")
    sc_light = sc_worldbody.findall("light")[1]

    joint_ranges = {}
    model_og = ET.parse(ASSETS_PATH + "mimo/MIMo_model.xml").getroot()
    for elem in model_og.iter():
        if elem.tag == "joint":
            joint_ranges[elem.attrib["name"]] = elem.attrib["range"]

    for i, age in enumerate(AGES):

        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        growth_params = calc_growth_params(age, SCENE_PATH)
        update_mimo(model, data, growth_params)

        height = sum([
            -model.body("left_upper_leg").pos[2],
            -model.body("left_lower_leg").pos[2],
            -model.body("left_foot").pos[2],
            model.geom("geom:left_foot2").size[2]
        ])
        model.body("hip").pos = [0, 0, height]
        mujoco.mj_forward(model, data)

        path_model = f"{ASSETS_PATH}mimo/MIMo_model_{i}.xml"
        mujoco.mj_saveLastXML(path_model, model)
        temp_files.append(path_model)

        model = ET.parse(path_model).getroot()
        mo_body = model.find("worldbody").find("body[@name='mimo_location']")
        mo_body = mo_body.find("body")

        for elem in mo_body.iter():
            if elem.tag == "joint":
                elem.attrib["range"] = joint_ranges[elem.attrib["name"]]

        mo_mujoco = ET.Element("mujoco")
        mo_mujoco.set("model", "MIMo")
        mo_mujoco.append(mo_body)

        for elem in mo_mujoco.iter():
            if "name" in elem.attrib.keys():
                elem.attrib["name"] = f"{elem.attrib['name']}_{i}"

        ET.ElementTree(mo_mujoco).write(
            path_model, encoding="utf-8", xml_declaration=True
        )

        meta_file = ET.parse(ASSETS_PATH + "mimo/MIMo_meta.xml").getroot()

        valid_keys = [
            "name", "joint1", "joint",
            "site", "geom1", "geom2",
            "body1", "body2"
        ]
        invalid_tags = ["material", "texture"]

        for elem in meta_file.iter():
            for key, val in elem.attrib.items():
                if key in valid_keys and elem.tag not in invalid_tags:
                    elem.attrib[key] = f"{val}_{i}"

        if i > 0:
            meta_file.remove(meta_file.find("default"))
            meta_file.remove(meta_file.find("asset"))

        meta_temp_path = f"{ASSETS_PATH}mimo/MIMo_meta_{i}.xml"
        ET.ElementTree(meta_file).write(
            meta_temp_path, encoding="utf-8", xml_declaration=True
        )
        temp_files.append(meta_temp_path)

        new_sc_body = copy.deepcopy(sc_body)
        new_sc_body.set("name", f"{i}")
        new_sc_body.set("pos", f"0 {i * 0.4} 0")  # next to each other
        # new_sc_body.set("pos", f"{i * 0.2} 0 0")  # in front of each other

        new_sc_body.find("freejoint").set("name", f"mimo_location_{i}")
        new_sc_body.find("include").set("file", f"mimo/MIMo_model_{i}.xml")

        new_sc_light = copy.deepcopy(sc_light)
        new_sc_light.set("target", f"upper_body_{i}")
        new_sc_light.set("diffuse", "0.17 0.17 0.17")

        sc_worldbody.append(new_sc_body)
        sc_worldbody.append(new_sc_light)

        new_sc_include_meta = copy.deepcopy(sc_include_meta)
        new_sc_include_meta.set("file", f"mimo/MIMo_meta_{i}.xml")
        scene.append(new_sc_include_meta)

    scene.remove(sc_include_meta)
    sc_worldbody.remove(sc_body)
    sc_worldbody.remove(sc_light)

    ET.ElementTree(scene).write(
        PATH_SCENE_TEMP, encoding="utf-8", xml_declaration=True
    )

    model = mujoco.MjModel.from_xml_path(PATH_SCENE_TEMP)
    data = mujoco.MjData(model)

    for file_ in temp_files:
        if os.path.exists(file_):
            os.remove(file_)

    model.body("growth_references").pos = [0, 0, -2]

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            pass


def strength_test(action: str = None, pos: str = "stand",
                  age: float = None, active: bool = False) -> None:
    """
    This function will perform a strength test with MIMo.
    In order to do this, the following steps are performed:
    - Adjust the age of MIMo
    - Bring MIMo into the specified starting position
    - Perform the given action by activating relevant actuators and
        enabling physics

    Arguments:
        - action (str): The action MIMo should perform. Default is none.
        - pos (str): The starting position of MIMo. This needs to match with
            the function `adjust_pos`. Default is 'stand'.
        - age (float): The age of MIMo. Default is none which means the
            original model is used.
        - active (bool): If the MuJoCo viewer should be active. This disables
            the action parameter. Default is false.
    """

    if age is not None:
        growth_model = adjust_mimo_to_age(age, SCENE_PATH, False)

    model = mujoco.MjModel.from_xml_path(growth_model)
    data = mujoco.MjData(model)

    model.body("growth_references").pos = [0, 0, -5]

    adjust_pos(pos, model, data)

    if age is not None:
        delete_growth_scene(growth_model)

    qpos_init = data.qpos.copy()
    qvel_init = data.qvel.copy()

    state = {"reset": False, "run": 0}

    def key_callback(keycode):
        if keycode == 32:  # space
            state["run"] += 1
        elif keycode == 341:  # strg
            state["reset"] = True

    if active:
        if action:
            print("The 'action' parameter has no effect in an active viewer.")
        with mujoco.viewer.launch(model, data) as viewer:
            while viewer.is_running():
                pass
        return

    args = {"model": model, "data": data, "key_callback": key_callback}
    with mujoco.viewer.launch_passive(**args) as viewer:
        while viewer.is_running():

            if not state["run"]:
                continue

            step_start = time.time()

            mujoco.mj_step(model, data)

            if state["reset"]:
                data.qpos[:], data.qvel[:] = qpos_init, qvel_init
                data.ctrl[:] = np.zeros(len(data.ctrl))
                mujoco.mj_forward(model, data)
                state["reset"], state["run"] = False, 0

            if state["run"] == 1 and action:
                if action == "lift_head" and pos in ["prone", "supine"]:
                    data.ctrl[4] = -1 if pos == "prone" else 1  # head_tilt
                elif action == "lift_chest" and pos == "prone":
                    data.ctrl[0] = -1  # hip_bend
                    data.ctrl[4] = -1  # head_tilt
                elif action == "lift_arms" and pos == "supine":
                    data.ctrl[12] = 1  # right_shoulder_horizontal
                    data.ctrl[13] = 1  # right_shoulder_ad_ab
                    data.ctrl[20] = 1  # left_shoulder_horizontal
                    data.ctrl[21] = 1  # left_shoulder_ad_ab
                elif action == "lift_legs" and pos == "supine":
                    data.ctrl[28] = -1  # right_hip_flex
                    data.ctrl[36] = -1  # left_hip_flex
                elif action == "lean" and "sit_" in pos:
                    sit_dir = pos.split("_")[1]
                    if sit_dir == "forward":
                        data.ctrl[0] = -1  # hip_bend (backwards)
                    elif sit_dir == "backward":
                        data.ctrl[0] = 1  # hip_bend  (forwards)
                    elif sit_dir == "left":
                        data.ctrl[2] = 1  # hip_lean (to the right)
                    elif sit_dir == "right":
                        data.ctrl[2] = -1  # hip_lean (to the left)
                elif action == "hold" and pos == "prone_on_arms":
                    data.ctrl[0] = -1  # hip_bend
                    data.ctrl[4] = -1  # head_tilt
                    data.ctrl[12] = 1  # right_shoulder_horizontal
                    # data.ctrl[15] = 0.2  # right_elbow
                    data.ctrl[20] = 1  # left_shoulder_horizontal
                    # data.ctrl[23] = 0.2  # left_elbow
                else:
                    print("Unknown configuration for 'pos' and 'action'.")
                state["run"] += 1

            viewer.sync()

            passed_time = (time.time() - step_start)
            time_until_next_step = model.opt.timestep - passed_time
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":

    func_map = {
        "growth": growth,
        "multiple_mimos": multiple_mimos,
        "strength_test": strength_test
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
        except NameError:
            kwargs[key] = value

    func_map[parser.parse_args().function](**kwargs)
