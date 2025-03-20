""" This module contains different plotting functions. """

from resources.mimoGrowth.constants import AGE_GROUPS
from resources.mimoGrowth.growth import adjust_mimo_to_age, delete_growth_scene
from resources.mimoGrowth.utils import load_measurements, store_base_values, \
    approximate_growth_functions
from resources.mimoGrowth.utils import growth_function as func
from resources.mimoGrowth.growth import calc_growth_params
from collections import defaultdict
import argparse
import os
import re
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import cv2


def growth_function(measurement: str = "head_circumference") -> None:
    """
    This function plots different growth functions with their
    associated original data points.

    Arguments:
        measurement (str): The body part the growth function belongs to.
            Default is 'head_circumference'.
    """

    age_samples = np.linspace(0, 24, 100)

    measurements = load_measurements()
    params = approximate_growth_functions(measurements)[measurement]
    pred = func(age_samples, *params)

    y_true = measurements[measurement]["mean"]
    y_pred = func(AGE_GROUPS, *params)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.plot(age_samples, pred, label="MIMo")
    plt.errorbar(
        AGE_GROUPS[:-1], measurements[measurement]["mean"][:-1],
        measurements[measurement]["std"][:-1],
        fmt="o", label="Original Data"
    )

    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")

    plt.legend()
    plt.show()


def all_growth_functions() -> None:
    """
    This function plots all growth function in a single plot.
    """

    age_samples = np.linspace(0, 24, 100)

    measurements = load_measurements()
    functions = approximate_growth_functions(measurements)

    i = 0
    for body_part in measurements:

        plt.subplot(4, 4, i + 1)

        params = functions[body_part]
        pred = func(age_samples, *params)

        label = body_part.replace("_", " ").title()
        plt.plot(age_samples, pred, label=label)
        plt.errorbar(
            AGE_GROUPS[:-1], measurements[body_part]["mean"][:-1],
            measurements[body_part]["std"][:-1],
            fmt="o", markersize=2,
        )

        plt.xlabel("Age (Months)", fontsize=8)
        plt.ylabel("Size (Centimeter)", fontsize=8)
        plt.legend(
            handlelength=0, handleheight=0, handletextpad=0,
            fontsize=8, loc='lower right'
        )

        i += 1

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def diff_func_type(type_: str) -> None:
    """
    This function plots a fitted growth function that is based
    on a different function type e.g. polynomial or splines.

    Arguments:
        type_ (str): Either 'poly' or 'spline'.
    """

    def growth_func(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    age_samples = np.linspace(0, 24, 100)

    measurement = load_measurements()["head_circumference"]

    x, y = AGE_GROUPS, measurement["mean"]

    if type_ == "poly":
        poly_params = curve_fit(growth_func, x, y)[0]
        pred = growth_func(age_samples, *poly_params)
    elif type_ == "spline":
        spline_func = CubicSpline(x, y)
        pred = spline_func(age_samples)

    plt.plot(age_samples, pred, label="Fitted Function")
    plt.errorbar(
        AGE_GROUPS[:-1], measurement["mean"][:-1],
        measurement["std"][:-1],
        fmt="o", label="Original Data"
    )

    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")

    plt.legend()
    plt.show()


def multiple_functions() -> None:
    """
    This function plots multiple growth functions to compare them.
    Just modify the below variable to select different functions.
    """

    body_parts_to_plot = [
        "ankle_circumference",
        "foot_length",
        "hip_breadth",
        "mid_thigh_circumference",
        "rump_knee_length",
        "shoulder_elbow_length",
    ]

    measurements = load_measurements()
    age_samples = np.linspace(0, 24, 100)

    for body_part in body_parts_to_plot:

        params = approximate_growth_functions(measurements)[body_part]
        pred = func(age_samples, *params)

        plt.plot(age_samples, pred, label=body_part.replace("_", " ").title())

    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")

    plt.legend()
    plt.show()


def density() -> None:
    """
    This functions plots the density of each geom.

    Note that identical geoms (e.g left_eye and right_eye) are only
    plotted once since they have the same density.
    """

    base_values = store_base_values("resources/mimoEnv/assets/growth.xml")

    names, densities = [], []

    for name, attributes in base_values["geom"].items():
        name = re.sub(r"geom:|left_|right_", "", name)
        if name not in names:
            names.append(name)
            densities.append(attributes["density"])

    plt.bar(names, densities, zorder=3, edgecolor="k")

    plt.xlabel("Geom")
    plt.ylabel("Density (kg/m³)")

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.25)

    plt.show()


def growth_param(
        metric: str = "mass", geoms: str = None, motors: str = None) -> None:
    """
    This function plots the development of mass or strength. It can either plot
    the average values of the growth of specific geoms/motors.

    Arguments:
        metric (str): Needs to be 'mass' or 'gear'.
        geom (str): The geom for which to plot the mass. Plot multiple geoms
            by separating them with a comma. Default is None, which plots the
            average mass of all geoms.
        motor (str): The motor for which to plot the gear value. Plot multiple
            motors by separating them with a comma. Default is None, which
            plots the average gear of all motors.
    """

    scene = "resources/mimoEnv/assets/growth.xml"
    ages = np.linspace(0, 24, 25)

    mass, avg_mass = [], []
    gear, avg_gear = [], []

    for age in ages:

        growth_params = calc_growth_params(age, scene)

        params_geoms = growth_params["geom"]
        params_motors = growth_params["motor"]

        if geoms:
            for geom in geoms.split(","):
                mass.append(params_geoms[geom]["mass"])
        if motors:
            for motor in motors.split(","):
                gear.append(params_motors[motor]["gear"])

        all_mass = [params_geoms[g]["mass"] for g in params_geoms]
        avg_mass.append(np.mean(all_mass))

        all_gear = [params_motors[m]["gear"] for m in params_motors]
        avg_gear.append(np.mean(all_gear))

    plt.xlabel("Age (Months)")
    plt.ylabel("Mass (kg)" if metric == "mass" else "Gear Value")

    y_mass = mass if geoms else avg_mass
    y_gear = gear if motors else avg_gear
    y = y_mass if metric == "mass" else y_gear

    label_mass = geom if geoms else "Average Mass"
    label_gear = motor if motors else "Average Gear"
    label = label_mass if metric == "mass" else label_gear
    plt.plot(ages, y, label=label)

    plt.legend()
    plt.show()


def comparison_who(metric: str = "height") -> None:
    """
    This function will compare a growth parameter of MIMo and
    real infants from WHO growth charts.

    Use this link for more information:
    https://www.cdc.gov/growthcharts/who-growth-charts.htm

    Arguments:
        metric (str): Selects which data will be compared. Can be one of the
            following: ['height', 'weight', 'bmi', 'head_circumference']
    """

    data = {"mimo": defaultdict(list), "WHO": {}}

    path = "resources/growth_charts/"
    for dirpath, _, filenames in os.walk(path):

        if filenames == []:
            continue

        dfs = []
        for path in filenames:
            full_path = os.path.join(dirpath, path)
            dfs.append(pd.read_excel(full_path))
        df = sum(dfs) / len(dfs)

        key = dirpath.split("/")[-1]
        data["WHO"][key] = df

    age_mimo = np.linspace(0, 24, 25)
    age_who = list(range(0, 25))

    for i, age in enumerate(age_mimo):

        print(f"{(i / len(age_mimo) * 100):.2f}%", end="\r")

        growth_scene = adjust_mimo_to_age(
            age, "resources/mimoEnv/assets/growth.xml", False)

        mj_model = mujoco.MjModel.from_xml_path(growth_scene)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        weight = mj_model.body("hip").subtreemass[0]
        data["mimo"]["weight"].append(weight)

        head_pos = mj_data.geom("head").xpos
        head_size = mj_model.geom("head").size
        height_head = head_pos[2] + head_size[0]
        foot_size = mj_model.geom("geom:left_foot3").size
        height_foot = mj_data.geom("geom:left_foot3").xpos[2] - foot_size[0]
        height = (height_head - height_foot) * 100
        data["mimo"]["height"].append(height)

        head_circum = mj_model.geom("head").size[0] * 2 * np.pi * 100
        data["mimo"]["head_circumference"].append(head_circum)

        bmi = weight / (((height - 0.7) / 100) ** 2)
        data["mimo"]["bmi"].append(bmi)

        delete_growth_scene(growth_scene)

    print("100.0%")

    plt.plot(age_mimo, data["mimo"][metric], label="MIMo")
    plt.errorbar(
        age_who, data["WHO"][metric]["M"][:25].tolist(),
        data["WHO"][metric]["M"][:25] * data["WHO"][metric]["S"][:25],
        linestyle="--", label="Mean with Standard Deviation"
    )
    plt.fill_between(
        age_who, data["WHO"][metric]["P5"][:25],
        data["WHO"][metric]["P95"][:25],
        color='gray', alpha=0.3,
        label="5th - 95th Percentile"
    )
    plt.fill_between(
        age_who, data["WHO"][metric]["P10"][:25],
        data["WHO"][metric]["P90"][:25],
        color='gray', alpha=0.4,
        label="10th - 90th Percentile"
    )

    y_label = {
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi": "Body Mass Index (BMI)",
        "head_circumference": "Head Circumference (cm)"
    }[metric]

    plt.xlabel("Age (months)")
    plt.ylabel(y_label)

    plt.legend()
    plt.show()


def tensorboard(
        experiment: str, metric: str, ages: list,
        start: str = "", show_sem: bool = True) -> None:
    """
    This function plots the average performance of a RL experiment based
    on the tensorboard events within the mimoEnv/models/ folder.

    Note that the illustrations.py file was used to perform the experiments
    and that this function only works for standup and roll_over tests.

    It is important that the experiments are saved in the following format:
    - Roll-Over: <starting_position>_age<t>_<version>
    - Standup: age<t>_<version>
    The starting position is either 'prone' or 'supine'. The age describes the
    age of MIMo when the experiment was done and the version can be anything
    that distinctly identifies experiments with the same settings. This is
    necessary to repeat an experiment multiple times and the plot the
    average performance.

    Arguments:
        experiment (str): Either 'standup' or 'roll_over'.
        metric (str): The metric from the tensorboard event to plot.
        ages (list): A list of MIMo's age that should be plotted.
        start (str): Only relevant for the roll_over experiment. Needs to be
            'prone' or 'supine'.
        show_sem (bool): If the standard error of the mean should be plotted.
    """

    base_path = r"C:\Users\miles\coding\MIMo\mimoEnv\models\tensorboard_logs"

    results = defaultdict(list)
    for age in ages:

        for root, _, _ in os.walk(os.path.join(base_path, experiment)):

            if any([word not in root for word in [start, f"age{age}", "PPO"]]):
                continue

            event_acc = EventAccumulator(root)
            event_acc.Reload()

            values = [e.value for e in event_acc.Scalars(metric)]
            results[age].append(values)

    x = np.array([e.step for e in event_acc.Scalars(metric)]) / 1000000
    for age in ages:

        y_mean = np.mean(results[age], 0)
        y_std = np.std(results[age], 0)
        y_sem = y_std / np.sqrt(len(results[age]))

        plt.plot(x, y_mean, label=f"{age} Month(s)")
        if show_sem:
            plt.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=0.25)

    plt.xlabel("Time Steps (Million)")
    plt.ylabel(metric.replace("rollout/", "").replace("_", " ").title())

    plt.legend()
    plt.show()


def video_to_image(
        path: str, overlay: bool = False,
        count_images: int = None, percentages: list = None) -> None:
    """
    This function takes a video and returns an image containing
    specific frames either as a sequence or as an overlay.

    Arguments:
        path (str): The path to the video.
        overlay (bool): If the image should be an overlay. Default is false.
        count_images (int): The amount of images in the sequence. This selects
            evenly distributed frames.
        percentages (list): Select specific frames based on percentages where
            0 means the first frame and 100 means the last frame.
    """

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if percentages:
        frame_indices = [
            int(p / 100 * (total_frames - 1)) for p in percentages]
    elif count_images is not None:
        frame_indices = np.linspace(
            0, total_frames - 1, count_images, dtype=int)
    else:
        raise ValueError("Either 'count_images' or 'percentages' must be set.")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if overlay:

        alpha = 0.5

        base_frame = frames[0].astype(float)
        for frame in frames[1:]:
            base_frame = cv2.addWeighted(
                base_frame, 1 - alpha, frame.astype(float), alpha, 0)

        plt.imshow(base_frame.astype(np.uint8))
        plt.axis("off")
        plt.show()

    else:

        _, axes = plt.subplots(1, len(frames), figsize=(12, 4))
        for ax, frame in zip(axes, frames):
            ax.imshow(frame)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    func_map = {
        "growth_function": growth_function,
        "all_growth_functions": all_growth_functions,
        "diff_func_type": diff_func_type,
        "multiple_functions": multiple_functions,
        "density": density,
        "growth_param": growth_param,
        "comparison_who": comparison_who,
        "tensorboard": tensorboard,
        "video_to_image": video_to_image,
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
