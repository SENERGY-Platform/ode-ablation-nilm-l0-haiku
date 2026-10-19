"""The operator: heat pump compressor power, estimated from the site main meter.

The main meter publishes its energy register, not its power. The operator turns the
register into the mean power of each 15-minute bucket, applies the transformer factor,
and predicts the compressor's share of that power from the bucket's power, the latest
outdoor temperature and the hour of day.

Three methods are yours. Everything else belongs to Operator Lib.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import BUCKET, train_model

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _utc(value) -> datetime.datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


def _bucket_floor(value: datetime.datetime) -> datetime.datetime:
    return value - (value - _EPOCH) % BUCKET


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400

    # Transformer ratio of the main meter. The register increments are multiplied by
    # this before they become power. Set by the developer; not verified against the
    # platform's own values yet.
    transformer_factor = 50.0

    # Longest gap between two register readings that still yields a bucket power.
    # A longer gap restarts the bucket count rather than spreading energy over it.
    max_gap_s = 900


class Operator(MLOperator):
    configType = CustomConfig

    # "main" carries the main meter's energy register, "weather" the outdoor
    # temperature. The selector name is what infer() receives.
    selectors = [
        Selector({"name": "main", "args": ["main_energy"]}),
        Selector({"name": "weather", "args": ["temperature"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing during
        # the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self.temperature: typing.Optional[float] = None
        self.last_ts: typing.Optional[datetime.datetime] = None
        self.last_energy: typing.Optional[float] = None
        # (bucket boundary, register value interpolated there) of the last boundary seen.
        self.prev_boundary: typing.Optional[typing.Tuple[datetime.datetime, float]] = None
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        A main meter reading closes at most one 15-minute bucket. The prediction for
        that bucket is returned with the bucket's start as its timestamp, so the value
        is about the interval it describes.
        """
        if selector == "weather":
            value = data.get("temperature")
            if value is not None:
                self.temperature = float(value)
            return None, None, None

        energy = data.get("main_energy")
        if energy is None:
            return None, None, None

        closed = self._close_buckets(float(energy), _utc(timestamp))
        if model is None or self.temperature is None or not closed:
            return None, None, None

        start, power_w = closed[-1]
        prediction = model.predict(
            {
                "power_w": power_w,
                "temperature_c": self.temperature,
                "hour": start.hour + start.minute / 60.0,
            }
        )
        return start, {"prediction": prediction}, None

    def _restart(self, ts: datetime.datetime, energy: float) -> None:
        self.last_ts = ts
        self.last_energy = energy
        self.prev_boundary = None

    def _close_buckets(self, energy: float, ts: datetime.datetime) -> typing.List[typing.Tuple[datetime.datetime, float]]:
        """Register reading in; closed buckets out, as (bucket start, mean watts)."""
        if self.last_ts is None:
            self._restart(ts, energy)
            return []
        if ts <= self.last_ts:
            return []
        if energy < self.last_energy or (ts - self.last_ts).total_seconds() > self.config.max_gap_s:
            # A counter reset or a gap: the bucket that spans it has no valid energy.
            self._restart(ts, energy)
            return []

        closed = []
        span = ts - self.last_ts
        boundary = _bucket_floor(self.last_ts) + BUCKET
        while boundary <= ts:
            fraction = (boundary - self.last_ts) / span
            value = self.last_energy + (energy - self.last_energy) * fraction
            if self.prev_boundary is not None and self.prev_boundary[0] == boundary - BUCKET:
                # kWh over 15 minutes -> mean watts: 1 kWh over 0.25 h is 4000 W.
                power_w = (value - self.prev_boundary[1]) * self.config.transformer_factor * 4000.0
                closed.append((boundary - BUCKET, power_w))
            self.prev_boundary = (boundary, value)
            boundary += BUCKET
        self.last_ts = ts
        self.last_energy = energy
        return closed

    def train(self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started, so params and metrics logged through "logger" land on the run the
        resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger, self.config.transformer_factor)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based to start with.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
