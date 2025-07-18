import metpy.calc
import numpy as np
from metpy.calc import cape_cin, parcel_profile
from metpy.units import units


def compute_cape(pressure: np.ndarray, temperature: np.ndarray, dewpoint: np.ndarray):
    flipped_presure = np.flip(pressure)
    flipped_temperature = np.flip(temperature)
    flipped_dewpoint = np.flip(dewpoint)
    profile = parcel_profile(
        flipped_presure * units("pascals"),
        flipped_temperature[0] * units("kelvin"),
        flipped_dewpoint[0] * units("kelvin"),
    )
    cape, cin = cape_cin(
        flipped_presure * units("pascals"),
        flipped_temperature * units("kelvin"),
        flipped_dewpoint * units("kelvin"),
        profile,
    )

    return cape.m, cin.m
