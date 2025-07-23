import metpy.calc
import metpy.units
from LSM.input_data import LSMInputData
import numpy as np
import metpy


def compute_dew_point(pressure, specific_humidity):
    """
    Compute dew point temperature (°C) from pressure (Pa) and specific humidity (kg/kg)

    Parameters:
        pressure_pa (float): Total air pressure in Pa
        specific_humidity (float): Specific humidity (dimensionless, e.g., 0.012 for 12 g/kg)

    Returns:
        float: Dew point temperature in °C
    """
    # Constants
    epsilon = 0.622  # Ratio of gas constants (dry air / water vapor)
    a = 6.112  # hPa
    b = 17.67
    c = 243.5  # °C

    r = specific_humidity / (1 - specific_humidity)
    e = (r * pressure / 100) / (epsilon + r)
    ln_ratio = np.log(e / a)

    Td = (c * ln_ratio) / (b - ln_ratio)

    return Td


def compute_additional_state(
    input_data: LSMInputData,
):
    # compute grid center pressure via linear interpolation
    center_pressure = np.zeros(
        (
            input_data.max_steps,
            input_data.edge_pressure[0].shape[0],
            input_data.edge_pressure[0].shape[1],
            input_data.edge_pressure[0].shape[2] - 1,
        )
    )
    for step in range(input_data.max_steps):
        for i in range(center_pressure.shape[1]):
            for j in range(center_pressure.shape[2]):
                for k in range(center_pressure.shape[3]):
                    center_pressure[step, i, j, k] = (
                        input_data.edge_pressure[step][i, j, k] + input_data.edge_pressure[step][i, j, k + 1]
                    ) / 2

    # compute temperature
    temperature = np.zeros(
        (
            input_data.max_steps,
            input_data.edge_pressure[0].shape[0],
            input_data.edge_pressure[0].shape[1],
            input_data.edge_pressure[0].shape[2] - 1,
        )
    )
    for step in range(input_data.max_steps):
        temperature[step, :, :, :] = input_data.potential_temperature[step]

    # compute dewpoint
    dewpoint = np.zeros(
        (
            input_data.max_steps,
            input_data.edge_pressure[0].shape[0],
            input_data.edge_pressure[0].shape[1],
            input_data.edge_pressure[0].shape[2] - 1,
        )
    )
    for step in range(input_data.max_steps):
        dewpoint[step, :, :, :] = (
            273.15
            + metpy.calc.dewpoint_from_specific_humidity(
                center_pressure[step, :, :, :] * metpy.units.units("pascals"),
                input_data.specific_humidity[step] * metpy.units.units("kg/kg"),
            ).m
        )

    return center_pressure, temperature, dewpoint
