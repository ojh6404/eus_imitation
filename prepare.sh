#!/usr/bin/bash

sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U setuptools wheel pip
python3.9 -m pip install -r requirements.txt
python3.9 -m pip install jax[cuda11_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3.9 -m pip install flax==0.7.5 optax==0.1.5 orbax
# python3.9 -m pip install tensorflow[and-cuda]
