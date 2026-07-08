# DEMIR

Install `uv` (https://docs.astral.sh/uv/) with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then run `uv sync` to install all dependencies in a virtual environment (requires Python 3.12.2 and CUDA 11.8). Activate it with `source .venv/bin/activate`.  

All experiment shell files are stored in `runs/full/` organised by environment and seed.  

To generate plots after training, run the scripts in `utils/` such as `python utils/plot_key_corridor.py`.  

Check `pyproject.toml` for the full list of dependencies; you may need to install PyTorch separately if your CUDA version differs.
