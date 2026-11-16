# wave-part-dual

One day, this repository will solve the universe; in the worst-case scenario — it won’t.

An open-ended playground for computer vision, machine learning, and experiments
that look educational, interesting, or sufficiently suspicious.

## Repository structure

```text
wave-part-dual/
├── notebooks/             # Explorations grouped by topic
│   └── classical_cv/      # Classical computer vision practice
├── src/wave_part_dual/    # Reusable code extracted from experiments
├── tests/                 # Tests for reusable code
├── data/                  # Local datasets; not committed
├── outputs/               # Generated results; not committed
└── _references/           # Private learning references; not committed
```

Notebook numbering is local to each topic. A future topic can therefore begin
at `01` without negotiating with every notebook that has ever existed.

Reusable code follows the same topic-first idea:

```text
src/wave_part_dual/
└── classical_cv/
    ├── masks.py          # Binary-mask measurements
    ├── camera.py         # Sensor and capture simulations
    ├── geometry.py       # Projection, distortion, rays, and distance
    └── calibration.py    # Synthetic evidence and camera estimation
```

New topic packages should be added only when experiments produce code useful
outside one notebook. Plotting that explains a particular lesson can stay next
to that lesson; not every cheerful rectangle needs a permanent API.

## Environment setup

The project targets CPython 3.12. Conda manages the local Python environment;
Python dependencies and their compatible ranges live in `pyproject.toml`.

```bash
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
python -m pytest
jupyter lab
```
