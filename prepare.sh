#!/usr/bin/bash

sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U setuptools wheel pip
python3.9 -m pip install -r requirements_octo.txt
python3.9 -m pip install jax[cuda12_pip]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3.9 -m pip install jaxlib==0.4.23 flax==0.7.5 optax==0.1.5 orbax
