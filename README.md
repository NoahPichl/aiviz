# ReadMe

## Getting Started

Check out `getting_startet.ipynb`

## Python Environments

The root project and TIME use separate environments because their NumPy
requirements are incompatible. Run these commands from the repository root:

```bash
uv sync
uv sync --project TIME --python 3.12
```

The environments are created at `.venv` and `TIME/.venv`, respectively.
Run root project commands with `uv run`, and TIME commands with
`uv run --project TIME`. For example:

```bash
uv run --project TIME python -c "import timebench; print(timebench.__file__)"
```

## TiRex-2 Experiments

Run the TiRex-2 experiment in the root environment, which includes its model
and benchmark dependencies. The script automatically loads `TIME_DATASET`
from the root `.env`:

```dotenv
TIME_DATASET=data/TIME
```

Run from the repository root so the relative dataset path resolves correctly:

```bash
uv run python TIME/experiments/tirex2_model.py \
	--dataset "Global_Influenza/W" \
	--terms short \
	--batch-size 16
```

Results are saved to `output/results/TiRex2` by default.

