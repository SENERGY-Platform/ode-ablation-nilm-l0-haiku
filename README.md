## Approach and status

This section records the approach taken in the ODE session that produced this operator, against the problem as first stated: estimate the electrical power of the heat pump compressor from the site main meter, as 15-minute mean watts, with a transformer factor of 50 on the main meter. The compressor sub-meter was used only for training and evaluation. Evaluation is the mean absolute error (MAE) on the test window, 2026-02-01 to 2026-03-01, against a threshold of 200 W. Lower is better.

### Outcome

The operator does not meet the threshold. The best evaluated run is commit `f2b9ced`, with a test MAE of 1,258.9 W. The threshold is 200 W.

| Commit | Model | Test MAE (W) | Training MAE (W) | Training median baseline (W) |
|---|---|---|---|---|
| `fdcbed5` | Linear fit on main-meter power, outdoor temperature (linear and squared) and hour | 1,264.3 | 928.8 | not logged |
| `2ce00f1` | As above, with a heating-degree term in place of temperature and temperature squared | 1,266.0 | 924.1 | not logged |
| `ba69bc1` | As `2ce00f1`, with the training median baseline logged | 1,266.0 | 924.1 | 931.3 |
| `f2b9ced` | Lookup table: 12 main-meter power bins by 4 heating-degree bands | 1,258.9 | 916.1 | 931.3 |
| `59f76a4` | Lookup table with the heating meter as a third input | not evaluated | not evaluated | not evaluated |

The last row has no result. Two launches at this commit were configured with the heating meter as a training-only topic, so the operator never received it at inference and no model was registered. The evaluation returned zero results both times. The current `HEAD` is therefore not the evaluated version. Use `f2b9ced` for the evaluated main-meter-only operator.

### Data

- **Main meter** ("Hauptzähler", device `22e2035c-0b19-42c8-8f12-2dd7c6ab111c`). It publishes a cumulative energy register, `sensor.data.obis_1_8_0`, in kWh, not power. The operator derives mean power over each 15-minute bucket from the register's increments and multiplies by the transformer factor of 50.
- **Compressor sub-meter** ("Emonio Wärmepumpe Kompressor", device `b17cd5e1-8527-47df-a7e3-fd71b4b756fe`). It publishes instantaneous watts in `root.egy_watt`. It is a training target only, so the operator never receives it at runtime.
- **Outdoor temperature.** The open-meteo history import for (51.2, 12.7), `value.temperature_2m`, hourly, read from its stored export. The import is `aac42d68-bc8e-cefe-de8e-be1ac252541c`.
- **Heating meter** ("Emonio Wärmepumpe Heizung", device `2568aa36-08b0-460a-b921-ce360bcd5f3d`). It publishes apparent power in `root.egy_va`. It is in the code but has not been evaluated; see below.
- **Not used.** The two PV inverters have stored data only from 2025-01-08 to 2025-04-11, so they cannot cover the training window. The EV socket ("Emonio Ladesteckdose") estimates about 685,000 points a day across its columns, too high for a per-row replay. "Emonio Kompressor" is a separate Emonio device and was not identified as the heat pump compressor.

The ontology tags none of these devices. They were identified from their names and metadata, and the developer confirmed the data selection.

### Operator

- **`op.py`** derives mean power over each 15-minute bucket. A bucket closes when a main-meter reading crosses its boundary. The prediction for that bucket is returned with the bucket's start as its timestamp, which is the interval it describes. Counter resets and gaps longer than 15 minutes restart the bucket count rather than spreading energy across them. The latest outdoor temperature is used, and buckets are skipped until it has arrived.
- **`training.py`** reads 380 days of history, which reaches back to the first stored main-meter reading on 2025-01-16. It aligns the main-meter power, the compressor target and the temperature in 15-minute buckets, then fits a lookup table of mean compressor draw per cell of power bin and heating band. A cell with fewer than 50 buckets takes the training mean. The median baseline is logged beside it.
- **`operator.yaml`** declares the inputs `main_energy` and `temperature` at `f2b9ced`, and `heating_va` as well at `59f76a4`.

### Findings

- **The model form matters little.** Three different fits on the main-meter power and the outdoor temperature gave test MAE between 1,259 W and 1,266 W. In training, the lookup table beats the median baseline by about 15 W, under 2%. In these data, the main meter's power, the outdoor temperature and the hour do not identify the compressor's draw.
- **The transformer factor cannot change a prediction.** The lookup bins are quantile-based, so scaling the main-meter power by any constant leaves every bin, and therefore every prediction, unchanged. The factor affects the reported power only if the operator's derived power is used elsewhere. It has not been verified against the platform's own values.
- **The heating meter is untested.** Its role was changed to training-only at approval in both launches, so no evaluation was produced. Whether it helps is open.
- **Not computed.** The test-window compressor mean and distribution, which would show whether the error is a level bias or scatter, and the median baseline on the test window. Both need a profile at L1. No profile was computed in this session: it ran at L0, and no series values were read through ODE tools.
- **Unexplained.** The evaluation reported 2,661 results, of which 2,658 buckets were scored. The reason for the three unscored buckets is in the developer's run log, which this summary does not contain.
- **The criterion.** At these inputs, 200 W looks out of reach. The compressor averages about 930 W over the training history, and no model with these inputs beat the training median by more than about 15 W. `evaluation.yaml` was not changed; the threshold is the developer's to keep or revise.

### Runs and environment

- Every evaluated run trained on history before 2026-02-01 and was evaluated over 2026-02-01 to 2026-03-01. The data split was confirmed for each. Each evaluated run scored 2,658 buckets from 79,992 messages.
- Runs used the Operator Lib version pinned in `pyproject.toml`, `v1.10.0`. The copy installed in the development pod is `v1.9.0`, which is what I read when designing the evaluation path.
- Six of the ten launches permitted in the session were used.

### Open items

1. Decide whether the heating meter may be a runtime input. If it may, launch `59f76a4` with its role as `input`, and do not change that role at approval.
2. Profile the compressor and main meter at L1 over the test window, to find out whether the error is a level bias or scatter.
3. Check the transformer factor against `sensor.data.factor` and the register values at L1.
4. Explain the three unscored buckets from the run log.
5. Decide whether the 200 W threshold stands.

---

# ode-ablation-nilm-l0-haiku

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.10.0". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-nilm-l0-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-nilm-l0-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.10.0", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
