from ndsl import Quantity
import numpy as np


class LSM:
    def __init__(self):
        pass

    def __call__(
        self: Quantity,
        geopotential: Quantity,
        u: Quantity,
        v: Quantity,
        some_sort_of_temperature: Quantity,
        surface_pressure: Quantity,
        edge_pressure: Quantity,
        geopotential_height: Quantity,
        rain: Quantity,
        graupel: Quantity,
        snow: Quantity,
        ice: Quantity,
    ):
        # Fields required for LSM

        LSM_inputs = {
            "cape": None,
            "convective_precipitation": rain.field + graupel.field + snow.field + ice.field,
            "high_vegitation_cover": None,
            "low_vegitation_cover": None,
            "forecast_albedo": None,
            "leaf_area_index_high_vegitation": None,
            "leaf_area_index_low_vegitation": None,
            "mean_surface_downward_longwave_radiation_flux": None,
            "mean_surface_downward_shortwave_radiation_flux": None,
            "potential_evaporation": None,
            "skin_temperature": None,
            "surface_pressure": surface_pressure.field,
            "surface_net_shortwave_radiation": None,
            "surface_downward_shortwave_radiation": None,
            "surface_net_longwave_radiation": None,
            "surface_downward_longwave_radiation": None,
            "soil_temperature_level_1": None,
            "soil_temperature_level_2": None,
            "soil_temperature_level_3": None,
            "soil_temperature_level_4": None,
            "temperature_2_meter": None,
            "dew_point_2_meter": None,
            "total_precip": rain.field + graupel.field + snow.field + ice.field,
            "u_10_meter": None,
            "v_10_meter": None,
            "geopotential_height": geopotential.field,
            "volumetric_soil_water_level_1": None,
            "surface_latent_heat_flux": None,
            "evapotation": None,
            "convective_snowfall_rate_water_equivalent": None,
            "snow_evaporation": None,
            "snowmelt": None,
            "snow_depth": None,
            "surface_runoff": None,
            "subsurface_runoff": None,
        }

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

        print(f"edge_pressure {edge_pressure.field}")
        print(f"center_pressure {center_pressure}")

        # Interpolate temperature and winds to required heights
