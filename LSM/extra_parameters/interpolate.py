import numpy as np


def interpolate_to_fixed_height(
    surface_geopotential: np.ndarray,
    geopotential_height_center: np.ndarray,
    data: np.ndarray,
    surface_data: float,
    target_height_above_surface: float,
):

    target_geopotential_height = target_height_above_surface + (surface_geopotential / 9.81)
    # print(f"Target geopotential height: {target_geopotential_height}")
    target_index = np.nan
    if target_geopotential_height > geopotential_height_center[0]:
        target_index = -999
    elif target_geopotential_height < geopotential_height_center[-1]:
        target_index = -1
    else:
        for k in range(len(geopotential_height_center) - 1):
            if (
                target_geopotential_height < geopotential_height_center[k]
                and target_geopotential_height > geopotential_height_center[k + 1]
            ):
                # print(
                #     f"Desired height of {target_geopotential_height} falls between {geopotential_height_center[k]} and {geopotential_height_center[k+1]}"
                # )
                target_index = k
                break

    if target_index == -999:
        # target height is above highest model level
        return np.nan
    if target_index == -1:
        # target height is between surface and center of lowest (nearest to surface) grid cell
        difference = 0 - geopotential_height_center[-1]
        distance_below = 0 - target_geopotential_height
        distance_above = target_geopotential_height - geopotential_height_center[-1]
        fraction_below = distance_below / difference
        fraction_above = distance_above / difference
        fraction = fraction_below + fraction_above  # should equal 1
        # if fraction != 1:
        #     raise ValueError(f"FRACTION VALUE {fraction} DOES NOT EQUAL ONE")
        return (data[-1] * fraction_above + surface_data * fraction_below) / 2
    else:
        # target height is between two layers
        difference = geopotential_height_center[target_index + 1] - geopotential_height_center[k]
        distance_below = geopotential_height_center[k + 1] - target_geopotential_height
        distance_above = target_geopotential_height - geopotential_height_center[k]
        fraction_below = distance_below / difference
        fraction_above = distance_above / difference
        fraction = fraction_below + fraction_above  # should equal 1
        # if fraction != 1:
        #     raise ValueError(f"FRACTION VALUE {fraction} DOES NOT EQUAL ONE")
        return (data[target_index] * fraction_above + data[target_index + 1] * fraction_below) / 2
