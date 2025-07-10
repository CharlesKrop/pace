from ndsl import Quantity
import numpy as np
from ndsl.grid import GridData
import tensorflow as tf


import numpy as np
from scipy.interpolate import interp1d


def interpolate_to_fixed_height(
    surface_geopotential: np.ndarray,
    geopotential_height: np.ndarray,
    data: np.ndarray,
    target_height: float,
    rank: float,
    method="linear",
):
    domain = data.shape
    interpolated_data = np.full(domain[0:2], np.nan)
    center_geopotential = np.full(domain, np.nan)

    for i in range(domain[0]):
        for j in range(domain[1]):
            target_geopotential = target_height + surface_geopotential[i, j]
            print(
                f"{rank, i, j} target height: {target_height}, surface geopotential {surface_geopotential[i, j]}, computed: {target_geopotential}"
            )
            for k in range(domain[2] - 1):
                if target_geopotential > geopotential_height[i, j, k]:
                    print(
                        f"Found target for {rank, i, j}, between {k} and {k+1}. Target: {target_geopotential}. Observed: {geopotential_height[i, j, k]}"
                    )
                    # target height is between current k and k+1
                    difference = geopotential_height[i, j, k + 1] - geopotential_height[i, j, k]
                    distance_from_upper = geopotential_height[i, j, k + 1] - target_geopotential
                    distance_from_lower = target_geopotential - geopotential_height[i, j, k]
                    fraction_upper = distance_from_upper / difference
                    fraction_lower = distance_from_lower / difference
                    fraction = fraction_upper + fraction_lower  # should equal 1

                    # linear interpolation of data to desired height
                    interpolated_data[i, j] = (
                        data[i, j, k + 1] * fraction_upper + data[i, j, k] * fraction_lower
                    ) / 2
                    # if fraction != 1:
                    #     print(
                    #         f"FRACTION VALUE: {fraction} at {i, j, k}. Target height: {target_geopotential}"
                    #     )
                    break

    return interpolated_data


class LSM:
    def __init__(self):
        # load the model
        self.sm_model = tf.keras.models.load_model("/Users/ckropiew/pace_llm/SM_for_GEOS/sm_model")

    def __call__(
        self,
        grid_data: GridData,
        surface_geopotential: Quantity,
        u: Quantity,
        v: Quantity,
        some_sort_of_temperature: Quantity,
        surface_pressure: Quantity,
        edge_pressure: Quantity,
        geopotential_height_center: Quantity,
        rain: Quantity,
        graupel: Quantity,
        snow: Quantity,
        ice: Quantity,
        rank: float,
    ):

        # Prepare pressure and temperature
        center_pressure = np.zeros(
            (edge_pressure.field.shape[0], edge_pressure.field.shape[1], edge_pressure.field.shape[2] - 1)
        )
        for i in range(center_pressure.shape[0]):
            for j in range(center_pressure.shape[1]):
                for k in range(center_pressure.shape[2]):
                    center_pressure[i, j, k] = (
                        edge_pressure.field[i, j, k] + edge_pressure.field[i, j, k + 1]
                    ) / 2

        # Interpolate temperature and winds to required heights
        self.interpolated_data = interpolate_to_fixed_height(
            surface_geopotential.field,
            geopotential_height_center.field,
            some_sort_of_temperature.field,
            2,
            rank,
        )

        # Fields required for LSM
        domain = u.field.shape
        junk_data = np.random.random(domain[0:2])

        LSM_input_vars = [
            "cape",
            "cp",
            "cvh",
            "cvl",
            "fal",
            "lai_hv",
            "lai_lv",
            "msdwlwrf",
            "msdwswrf",
            "pev",
            "skt",
            "sp",
            "ssr",
            "ssrd",
            "str",
            "strd",
            "stl1",
            "stl2",
            "stl3",
            "stl4",
            "t2m",
            "d2m",
            "tp",
            "u10",
            "v10",
            "z",
            "swvl1",
            "slhf",
            "e",
            "csfr",
            "es",
            "smlt",
            "sd",
            "sro",
            "ssro",
        ]

        for i in range(domain[0]):
            for j in range(domain[1]):
                LSM_inputs = {}
                batch_size = 2
                past_times = 48
                for var in LSM_input_vars:
                    array = np.full((batch_size, past_times, 1, 1), np.nan)
                    for batch in range(batch_size):
                        for time in range(past_times):
                            if var == "cp":
                                array[batch, time, 0, 0] = (
                                    rain.field[i, j, -1]
                                    + graupel.field[i, j, -1]
                                    + snow.field[i, j, -1]
                                    + ice.field[i, j, -1]
                                )
                            elif var == "sp":
                                array[batch, time, 0, 0] = surface_pressure.field[i, j]
                            elif var == "tp":
                                array[batch, time, 0, 0] = (
                                    rain.field[i, j, -1]
                                    + graupel.field[i, j, -1]
                                    + snow.field[i, j, -1]
                                    + ice.field[i, j, -1]
                                )
                            elif var == "z":
                                array[batch, time, 0, 0] = surface_geopotential.field[i, j]
                            else:
                                array[batch, time, 0, 0] = junk_data[i, j]
                        LSM_inputs[var] = array

                print(f"README {type(LSM_inputs), LSM_inputs.keys()}")
                for var in LSM_inputs.keys():
                    print(f"{var}: {LSM_inputs[var].shape}")
                # Run the model
                # self.sm_model.predict_on_batch(LSM_inputs)
