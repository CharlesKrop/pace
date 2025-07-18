import xarray as xr


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
