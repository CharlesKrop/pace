from collections import deque
from ndsl import Quantity
import numpy as np


class LSMInputData:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.surface_geopotential = deque(maxlen=max_steps)
        self.u = deque(maxlen=max_steps)
        self.v = deque(maxlen=max_steps)
        self.potential_temperature = deque(maxlen=max_steps)
        self.specific_humidity = deque(maxlen=max_steps)
        self.surface_pressure = deque(maxlen=max_steps)
        self.edge_pressure = deque(maxlen=max_steps)
        self.kappa_pressure = deque(maxlen=max_steps)
        self.geopotential_height_center = deque(maxlen=max_steps)
        self.geopotential_height_interface = deque(maxlen=max_steps)
        self.rain = deque(maxlen=max_steps)
        self.graupel = deque(maxlen=max_steps)
        self.snow = deque(maxlen=max_steps)
        self.ice = deque(maxlen=max_steps)
        self.rank = deque(maxlen=max_steps)

    def add_data(
        self,
        surface_geopotential: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        potential_temperature: np.ndarray,
        specific_humidity: np.ndarray,
        surface_pressure: np.ndarray,
        edge_pressure: np.ndarray,
        kappa_pressure: np.ndarray,
        geopotential_height_center: np.ndarray,
        geopotential_height_interface: np.ndarray,
        rain: np.ndarray,
        graupel: np.ndarray,
        snow: np.ndarray,
        ice: np.ndarray,
    ):

        self.surface_geopotential.append(surface_geopotential)
        self.u.append(u)
        self.v.append(v)
        self.potential_temperature.append(potential_temperature)
        self.specific_humidity.append(specific_humidity)
        self.surface_pressure.append(surface_pressure)
        self.edge_pressure.append(edge_pressure)
        self.kappa_pressure.append(kappa_pressure)
        self.geopotential_height_center.append(geopotential_height_center)
        self.geopotential_height_interface.append(geopotential_height_interface)
        self.rain.append(rain)
        self.graupel.append(graupel)
        self.snow.append(snow)
        self.ice.append(ice)
