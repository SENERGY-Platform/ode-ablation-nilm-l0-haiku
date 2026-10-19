"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The history is read with provide_historic_data(). The main meter and the weather are
input topics; the compressor sub-meter is a training topic, so the operator learns
from it and never receives it at runtime.

The model is a lookup table: the mean compressor draw in each cell of main-meter power
(quantile bins) crossed with a band of outdoor temperature (heating degrees).
"""

import datetime
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads. The main meter has stored history from
# 2025-01-16 and the split is 2026-02-01, so 380 days reaches back to its first reading.
TRAINING_WINDOW = datetime.timedelta(days=380)

# The bucket the target is averaged over. The evaluation uses the same 15 minutes.
BUCKET = datetime.timedelta(minutes=15)

_BUCKET_FREQ = "15min"
_MAX_GAP = pd.Timedelta(minutes=15)
_MIN_COVERAGE = pd.Timedelta(minutes=12)
_MAX_WEATHER_AGE = pd.Timedelta(hours=6)
_MIN_BUCKETS = 100

# Outdoor temperature at and above which there is no heating demand, in degrees Celsius.
_HEATING_BASE_C = 18.0

# Power is split into this many quantile bins of the training data.
_POWER_BINS = 12

# A cell with fewer training buckets than this takes the training mean instead.
_MIN_CELL = 50

# kWh of register per 15-minute bucket to mean watts: 1 kWh over 0.25 h is 4000 W.
_WATTS_PER_KWH_PER_BUCKET = 4000.0


def _band(temperature_c: np.ndarray) -> np.ndarray:
    """Heating band of each temperature: 0 none, 1 up to 5, 2 up to 10, 3 above 10 degrees below base."""
    heating_degrees = np.maximum(0.0, _HEATING_BASE_C - np.asarray(temperature_c, dtype=float))
    return np.select(
        [heating_degrees <= 0.0, heating_degrees <= 5.0, heating_degrees <= 10.0],
        [0, 1, 2],
        default=3,
    ).astype(int)


class CompressorPowerModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives exactly the payload infer() builds, so the two are one contract
    in two files. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, power_edges: typing.List[float], table: typing.List[typing.List[float]]) -> None:
        self.power_edges = [float(e) for e in power_edges]
        self.table = [[float(v) for v in row] for row in table]

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        power = float(payload["power_w"])
        temperature = float(payload["temperature_c"])
        bin_index = int(np.searchsorted(np.asarray(self.power_edges, dtype=float), power, side="right"))
        band = int(_band(np.array([temperature]))[0])
        return float(self.table[band][bin_index])


def _as_frames(refs) -> typing.List[pd.DataFrame]:
    frames = []
    for ref in refs:
        dataset = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        frames.append(dataset.to_pandas())
    return frames


def _register_power(frame: pd.DataFrame, factor: float) -> pd.Series:
    """Mean watts per 15-minute bucket, from the main meter's energy register."""
    f = frame.dropna(subset=["main_energy"]).sort_values("time").reset_index(drop=True)
    if len(f) < 2:
        return pd.Series(dtype=float, name="power_w")
    t = pd.to_datetime(f["time"])
    energy = f["main_energy"].astype(float)
    increment = energy.diff()
    gap = t.diff()
    ok = (increment >= 0) & (gap <= _MAX_GAP)
    rows = pd.DataFrame({"t": t, "inc": increment.where(ok, 0.0), "bad": ~ok}).iloc[1:]
    rows["bucket"] = rows["t"].dt.floor(_BUCKET_FREQ)
    g = rows.groupby("bucket").agg(inc=("inc", "sum"), bad=("bad", "max"), last=("t", "max"))
    covered = (g["last"] - g.index.to_series()) >= _MIN_COVERAGE
    good = (~g["bad"].astype(bool)) & covered
    return (g.loc[good, "inc"] * factor * _WATTS_PER_KWH_PER_BUCKET).rename("power_w")


def _target(frame: pd.DataFrame) -> pd.Series:
    f = frame.dropna(subset=["compressor_w"])
    t = pd.to_datetime(f["time"])
    return f.groupby(t.dt.floor(_BUCKET_FREQ))["compressor_w"].mean().rename("target_w")


def _weather(frame: pd.DataFrame) -> pd.Series:
    f = frame.dropna(subset=["temperature"])
    s = pd.Series(f["temperature"].astype(float).to_numpy(), index=pd.to_datetime(f["time"]))
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")]


def train_model(logger: TrainMlflowLogger, factor: float) -> typing.Optional[PythonModel]:
    """Read the history, align it in 15-minute buckets, fit a lookup table, and hand back a model."""
    with logger.trace("read history"):
        inputs = _as_frames(provide_historic_data(TRAINING_WINDOW))
        targets = _as_frames(provide_historic_data(TRAINING_WINDOW, role="training"))

    main = next((f for f in inputs if "main_energy" in f.columns), None)
    weather = next((f for f in inputs if "temperature" in f.columns), None)
    target = next((f for f in targets if "compressor_w" in f.columns), None)
    if main is None or weather is None or target is None:
        # Explicitly nothing rather than a model fitted on partial data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("align buckets"):
        data = pd.concat([_register_power(main, factor), _target(target)], axis=1, join="inner").dropna()
        temperature = _weather(weather).reindex(data.index, method="ffill", tolerance=_MAX_WEATHER_AGE)
        data["temperature_c"] = temperature.to_numpy()
        data = data.dropna()

    if len(data) < _MIN_BUCKETS:
        return None

    with logger.trace("fit"):
        power = data["power_w"].to_numpy(dtype=float)
        temperature_c = data["temperature_c"].to_numpy(dtype=float)
        y = data["target_w"].to_numpy(dtype=float)

        edges = np.unique(np.quantile(power, np.linspace(0.0, 1.0, _POWER_BINS + 1)[1:-1]))
        n_bins = len(edges) + 1
        bins = np.searchsorted(edges, power, side="right")
        bands = _band(temperature_c)

        sums = np.zeros((4, n_bins))
        counts = np.zeros((4, n_bins))
        np.add.at(sums, (bands, bins), y)
        np.add.at(counts, (bands, bins), 1.0)
        fallback = float(np.mean(y))
        table = np.where(counts >= _MIN_CELL, sums / np.maximum(counts, 1.0), fallback)

        predicted = table[bands, bins]
        mae = float(np.mean(np.abs(predicted - y)))
        # The reference the model has to beat: predicting the training median everywhere,
        # on the same buckets.
        constant_mae = float(np.mean(np.abs(y - np.median(y))))

    logger.log_params(
        {
            "training_window_days": TRAINING_WINDOW.days,
            "transformer_factor": factor,
            "training_buckets": len(data),
            "power_bins": n_bins,
            "heating_bands": 4,
            "min_cell_buckets": _MIN_CELL,
        }
    )
    logger.log_metrics(
        {
            "train_mae_binned_w": mae,
            "train_mae_constant_w": constant_mae,
            "train_mean_target_w": fallback,
        }
    )
    return CompressorPowerModel(edges.tolist(), table.tolist())
