import xarray as xr
from LSM.input_data import LSMInputData


def normalize_batch(data, summary):

    stats = {
        k: [
            float(summary[k].to_dataset("statistics")["min"]),
            float(summary[k].to_dataset("statistics")["max"]),
            float(summary[k].to_dataset("statistics")["median"]),
            float(summary[k].to_dataset("statistics")["p25"]),
            float(summary[k].to_dataset("statistics")["p75"]),
        ]
        for k in summary.keys()
    }

    for k, v in stats.items():
        stats[k].append((v[0] - v[2]) / (v[4] - v[3] + 1e-10))
        stats[k].append((v[1] - v[2]) / (v[4] - v[3] + 1e-10))

    # robust and normalize for es
    data = {
        k: (
            (v - stats[k][0]) / (stats[k][1] - stats[k][0] + 1e-10)
            if k == "es"
            else (v - stats[k][2]) / (stats[k][4] - stats[k][3] + 1e-10)
        )
        for k, v in data.items()
    }
    # normalize robust for all but es
    data = {
        k: v if k == "es" else (v - stats[k][5]) / (stats[k][6] - stats[k][5] + 1e-10)
        for k, v in data.items()
    }
    # scale normalize
    data = {k: v * (1 - (-1)) + (-1) for k, v in data.items()}

    return data


def normalize_lsm_inputs(data: dict, summary: xr.Dataset):
    stats = {}

    # Step 1: Extract key statistics for each variable
    for key in summary:
        print(key)
        stats_ds = summary[key].to_dataset("statistics")
        stats[key] = {
            "min": float(stats_ds["min"]),
            "max": float(stats_ds["max"]),
            "median": float(stats_ds["median"]),
            "p25": float(stats_ds["p25"]),
            "p75": float(stats_ds["p75"]),
        }

        # Add robust min/max for normalization
        inner_quartile_range = stats[key]["p75"] - stats[key]["p25"] + 1e-10  # prevent divide by zero
        stats[key]["robust_min"] = (stats[key]["min"] - stats[key]["median"]) / inner_quartile_range
        stats[key]["robust_max"] = (stats[key]["max"] - stats[key]["median"]) / inner_quartile_range

    # Step 2: Initial normalization
    normalized_data = {}
    for key, values in data.items():
        s = stats[key]

        if key == "es":
            # Standard min-max normalization
            norm = (values - s["min"]) / (s["max"] - s["min"] + 1e-10)
        else:
            # Robust normalization using median and IQR
            norm = (values - s["median"]) / (s["p75"] - s["p25"] + 1e-10)

        normalized_data[key] = norm

    # Step 3: Re-normalize using robust min/max (except 'es')
    for key, values in normalized_data.items():
        if key == "es":
            continue

        s = stats[key]
        norm = (values - s["robust_min"]) / (s["robust_max"] - s["robust_min"] + 1e-10)
        normalized_data[key] = norm

    # Step 4: Rescale everything to [-1, 1]
    for key, values in normalized_data.items():
        normalized_data[key] = values * 2 - 1  # maps [0,1] to [-1,1]

    return normalized_data
