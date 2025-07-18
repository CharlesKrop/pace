from ndsl import Quantity, StencilFactory
import numpy as np
from ndsl.grid import GridData
import tensorflow as tf
import xarray as xr
import numpy as np
from LSM.input_data import LSMInputData
from LSM.extra_parameters.CAPE import compute_cape
from LSM.extra_parameters.interpolate import interpolate_to_fixed_height
from LSM.normalize_batch import normalize_batch
from LSM.extra_parameters.base_state import compute_additional_state


def update_qvapor(stencil_factory: StencilFactory, soil_moisture: np.ndarray, qvapor: Quantity):
    for i in range(stencil_factory.grid_indexing.domain[0]):
        for j in range(stencil_factory.grid_indexing.domain[1]):
            # bring qvapor at the surface to halfway between the old qvapor value and the soil moisture value
            qvapor.field[i, j, -1] = (
                qvapor.field[i, j, -1] - (qvapor.field[i, j, -1] - soil_moisture[i, j]) / 2
            )

        return qvapor


class LSM:
    def __init__(self, grid_data: GridData, stencil_factory: StencilFactory):
        self.grid_data = grid_data
        self.stencil_factory = stencil_factory

        # load the model
        self.sm_model = tf.keras.models.load_model("./LSM/sm_model")
        self.stats = xr.open_dataset("./LSM/stats.nc")

        # load ERA5 data
        self.era5_accum = xr.open_dataset("./LSM/extra_parameters/era5_data_Type-accum.nc")
        self.era5_avg = xr.open_dataset("./LSM/extra_parameters/era5_data_Type-avg.nc")
        self.era5_instant = xr.open_dataset("./LSM/extra_parameters/era5_data_Type-instant.nc")

    def __call__(
        self,
        input_data: LSMInputData,
        rank: float,
    ):
        import os

        directory = "./LSM/debug_data"

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except:
                continue

        # Fields required for LSM
        domain = self.stencil_factory.grid_indexing.domain
        junk_data = np.random.random(domain[0:2])

        center_pressure, temperature, dewpoint = compute_additional_state(input_data)

        LSM_input_vars = [
            "cape",  # Convective Available Potential Energy
            "cp",  # Convective Precipitation
            "cvh",  # High vegetation cover
            "cvl",  # Low vegetation cover
            "fal",  # Forecast albedo
            "lai_hv",  # Leaf area index, high vegetation
            "lai_lv",  # Leaf area index, low vegetation
            "msdwlwrf",  # Mean surface downward long-wave radiation flux
            "msdwswrf",  # Mean surface downward short-wave radiation flux
            "pev",  # potential evaporation
            "skt",  # Skin temperature
            "sp",  # Surface pressure
            "ssr",  # Surface net short-wave (solar) radiation
            "ssrd",  # Surface short-wave (solar) radiation downwards
            "str",  # Surface net long-wave (thermal) radiation
            "strd",  # Surface long-wave (thermal) radiation downwards
            "stl1",  # Soil temperature level 1
            "stl2",  # Soil temperature level 2
            "stl3",  # Soil temperature level 3
            "stl4",  # Soil temperature level 4
            "t2m",  # 2 metre temperature
            "d2m",  # 2 metre dewpoint temperature
            "tp",  # Total precipitation
            "u10",  # 10 metre U wind component
            "v10",  # 10 metre V wind component
            "z",  # Geopotential
            "swvl1",  # Volumetric soil water layer 1
            "slhf",  # surface latent heat flux
            "e",  # Evaporation
            "csfr",  # Convective snowfall rate water equivalent
            "es",  # Snow evaporation
            "smlt",  # Snowmelt
            "sd",  # Snow Depth
        ]

        self.soil_moisture = np.full((domain[0], domain[1]), np.nan)
        self.soil_moisture_normalized = np.full((domain[0], domain[1]), np.nan)
        for i in range(domain[0]):
            for j in range(domain[1]):
                print(f"START {i, j}", flush=True)
                LSM_inputs = {}
                batch_size = 1
                past_times = 48

                # find nearest lat lon point for ERA5 data
                lat_index = np.abs(
                    self.era5_accum["latitude"].values - self.grid_data.lat_agrid.field[i, j]
                ).argmin()
                lon_index = np.abs(
                    self.era5_accum["longitude"].values - self.grid_data.lon_agrid.field[i, j]
                ).argmin()

                # fill inputs for the LSM
                for var in LSM_input_vars:
                    array = np.full((batch_size, past_times, 1, 1), np.nan)
                    for batch in range(batch_size):
                        for time in range(past_times):
                            if var == "cape":
                                if time < input_data.max_steps:
                                    cape, _ = compute_cape(
                                        center_pressure[time, i, j, :],
                                        temperature[time, i, j, :],
                                        dewpoint[time, i, j, :],
                                    )
                                    array[batch, time, 0, 0] = cape
                                else:
                                    cape, _ = compute_cape(
                                        center_pressure[-1, i, j, :],
                                        temperature[-1, i, j, :],
                                        dewpoint[-1, i, j, :],
                                    )
                                    array[batch, time, 0, 0] = cape
                            if var == "cp":
                                if time < input_data.max_steps:
                                    array[batch, time, 0, 0] = (
                                        input_data.rain[time].field[i, j, -1]
                                        + input_data.graupel[time].field[i, j, -1]
                                        + input_data.snow[time].field[i, j, -1]
                                        + input_data.ice[time].field[i, j, -1]
                                    )
                                else:
                                    array[batch, time, 0, 0] = (
                                        input_data.rain[-1].field[i, j, -1]
                                        + input_data.graupel[-1].field[i, j, -1]
                                        + input_data.snow[-1].field[i, j, -1]
                                        + input_data.ice[-1].field[i, j, -1]
                                    )
                            elif var == "cvh":
                                array[batch, time, 0, 0] = self.era5_instant["cvh"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "cvl":
                                array[batch, time, 0, 0] = self.era5_instant["cvl"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "fal":
                                array[batch, time, 0, 0] = self.era5_instant["fal"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "lai_hv":
                                array[batch, time, 0, 0] = self.era5_instant["lai_hv"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "lai_lv":
                                array[batch, time, 0, 0] = self.era5_instant["lai_lv"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "msdwlwrf":
                                array[batch, time, 0, 0] = self.era5_avg["avg_sdlwrf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "msdwswrf":
                                array[batch, time, 0, 0] = self.era5_avg["avg_sdswrf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "pev":
                                array[batch, time, 0, 0] = self.era5_accum["pev"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "skt":
                                array[batch, time, 0, 0] = self.era5_instant["skt"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "sp":
                                if time < input_data.max_steps:
                                    array[batch, time, 0, 0] = input_data.edge_pressure[time].field[i, j, -1]
                                else:
                                    array[batch, time, 0, 0] = input_data.edge_pressure[-1].field[i, j, -1]
                            elif var == "ssr":
                                array[batch, time, 0, 0] = self.era5_avg["avg_snswrf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "ssrd":
                                array[batch, time, 0, 0] = self.era5_accum["ssrd"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "str":
                                array[batch, time, 0, 0] = self.era5_avg["avg_snlwrf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "strd":
                                array[batch, time, 0, 0] = self.era5_accum["strd"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "stl1":
                                array[batch, time, 0, 0] = self.era5_instant["stl1"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "stl2":
                                array[batch, time, 0, 0] = self.era5_instant["stl2"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "stl3":
                                array[batch, time, 0, 0] = self.era5_instant["stl3"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "stl4":
                                array[batch, time, 0, 0] = self.era5_instant["stl4"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "t2m":
                                if time < input_data.max_steps:
                                    # TODO figure out how to normalize to negative geopotential heights
                                    # TODO get real surface data instead of using lowest grid center data
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[time].field[i, j],
                                        input_data.geopotential_height_center[time].field[i, j, :],
                                        temperature[time, i, j, :],
                                        temperature[time, i, j, -1],  # TODO need actual surface data
                                        2,
                                    )
                                else:
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[-1].field[i, j],
                                        input_data.geopotential_height_center[-1].field[i, j, :],
                                        temperature[-1, i, j, :],
                                        temperature[-1, i, j, -1],  # TODO need actual surface data
                                        2,
                                    )
                            elif var == "d2m":
                                if time < input_data.max_steps:
                                    # TODO figure out how to normalize to negative geopotential heights
                                    # TODO get real surface data instead of using lowest grid center data
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[time].field[i, j],
                                        input_data.geopotential_height_center[time].field[i, j, :],
                                        dewpoint[time, i, j, :],
                                        dewpoint[time, i, j, -1],  # TODO need actual surface data
                                        2,
                                    )
                                else:
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[-1].field[i, j],
                                        input_data.geopotential_height_center[-1].field[i, j, :],
                                        dewpoint[-1, i, j, :],
                                        dewpoint[-1, i, j, -1],  # TODO need actual surface data
                                        2,
                                    )
                            elif var == "tp":
                                if time < input_data.max_steps:
                                    array[batch, time, 0, 0] = (
                                        input_data.rain[time].field[i, j, -1]
                                        + input_data.graupel[time].field[i, j, -1]
                                        + input_data.snow[time].field[i, j, -1]
                                        + input_data.ice[time].field[i, j, -1]
                                    )
                                else:
                                    array[batch, time, 0, 0] = (
                                        input_data.rain[-1].field[i, j, -1]
                                        + input_data.graupel[-1].field[i, j, -1]
                                        + input_data.snow[-1].field[i, j, -1]
                                        + input_data.ice[-1].field[i, j, -1]
                                    )
                            elif var == "u10":
                                if time < input_data.max_steps:
                                    # TODO figure out how to normalize to negative geopotential heights
                                    # TODO get real surface data instead of using lowest grid center data
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[time].field[i, j],
                                        input_data.geopotential_height_center[time].field[i, j, :],
                                        input_data.u[time].field[i, j, :],
                                        input_data.u[time].field[i, j, -1],  # TODO need actual surface data
                                        10,
                                    )
                                else:
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[-1].field[i, j],
                                        input_data.geopotential_height_center[-1].field[i, j, :],
                                        input_data.u[-1].field[i, j, :],
                                        input_data.u[-1].field[i, j, -1],  # TODO need actual surface data
                                        10,
                                    )
                            elif var == "v10":
                                if time < input_data.max_steps:
                                    # TODO figure out how to normalize to negative geopotential heights
                                    # TODO get real surface data instead of using lowest grid center data
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[time].field[i, j],
                                        input_data.geopotential_height_center[time].field[i, j, :],
                                        input_data.v[time].field[i, j, :],
                                        input_data.v[time].field[i, j, -1],  # TODO need actual surface data
                                        10,
                                    )
                                else:
                                    array[batch, time, 0, 0] = interpolate_to_fixed_height(
                                        input_data.surface_geopotential[-1].field[i, j],
                                        input_data.geopotential_height_center[-1].field[i, j, :],
                                        input_data.v[-1].field[i, j, :],
                                        input_data.v[-1].field[i, j, -1],  # TODO need actual surface data
                                        10,
                                    )
                            elif var == "z":
                                if time < input_data.max_steps:
                                    array[batch, time, 0, 0] = input_data.surface_geopotential[time].field[
                                        i, j
                                    ]
                                else:
                                    array[batch, time, 0, 0] = input_data.surface_geopotential[-1].field[i, j]
                            elif var == "swvl1":
                                array[batch, time, 0, 0] = self.era5_instant["swvl1"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "slhf":
                                array[batch, time, 0, 0] = self.era5_accum["slhf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "e":
                                array[batch, time, 0, 0] = self.era5_accum["slhf"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "csfr":
                                array[batch, time, 0, 0] = self.era5_instant["csfr"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "es":
                                array[batch, time, 0, 0] = self.era5_accum["es"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "smlt":
                                array[batch, time, 0, 0] = self.era5_accum["smlt"].values[
                                    0, lat_index, lon_index
                                ]
                            elif var == "sd":
                                array[batch, time, 0, 0] = self.era5_instant["sd"].values[
                                    0, lat_index, lon_index
                                ]
                        LSM_inputs[var] = array

                # Normalize data
                LSM_inputs_normalized = normalize_batch(LSM_inputs, self.stats)

                # Run the model with un-normalized data
                self.soil_moisture[i, j] = self.sm_model.predict_on_batch(LSM_inputs)["soil_moisture"][0][0]
                # Run the model with normalized data
                self.soil_moisture_normalized[i, j] = self.sm_model.predict_on_batch(LSM_inputs_normalized)[
                    "soil_moisture"
                ][0][0]
                print(f"END {i, j}", flush=True)
