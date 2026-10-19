"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The history is read with provide_historic_data(). The main meter and the weather are
input topics; the compressor sub-meter is a training topic, so the operator learns
from it and never receives it at runtime.
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

# kWh of register per 15-minute bucket to mean watts: 1 kWh over 0.25 h is 4000 W.
_WATTS_PER_KWH_PER_BUCKET = 4000.0


def design(power_w, temperature_c, hour) -> np.ndarray:
    """The feature matrix, shared by training and by the model's predict().

    Power enters in kW, temperature in degrees Celsius, hour of day in UTC.
    """
    power_kw = np.asarray(power_w, dtype=float) / 1000.0
    temperature = np.asarray(temperature_c, dtype=float)
    angle = 2.0 * np.pi * np.asarray(hour, dtype=float) / 24.0
    return np.column_stack(
        [
            np.ones_like(power_kw),
            power_kw,
            temperature,
            temperature * temperature,
            np.sin(angle),
            np.cos(angle),
        ]
    )


class CompressorPowerModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives exactly the payload infer() builds, so the two are one contract
    in two files. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, coefficients: typing.List[float]) -> None:
        self.coefficients = [float(c) for c in coefficients]

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        x = design([float(payload["power_w"])], [float(payload["temperature_c"])], [float(payload["hour"])])[0]
        return max(0.0, float(x @ np.array(self.coefficients)))


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
    """Read the history, align it in 15-minute buckets, fit, and hand back a model."""
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
        hour = np.asarray(data.index.hour) + np.asarray(data.index.minute) / 60.0
        x = design(data["power_w"].to_numpy(), data["temperature_c"].to_numpy(), hour)
        y = data["target_w"].to_numpy()
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        mae = float(np.mean(np.abs(np.maximum(x @ coefficients, 0.0) - y)))

    logger.log_params(
        {
            "training_window_days": TRAINING_WINDOW.days,
            "transformer_factor": factor,
            "training_buckets": len(data),
        }
    )
    logger.log_metrics({"train_mae_w": mae, "train_mean_target_w": float(np.mean(y))})
    return CompressorPowerModel(coefficients.tolist())
