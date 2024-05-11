#!/usr/bin/bash

sudo apt install -y python3.9 python3.9-dev python3.9-venv
git clone https://github.com/ojh6404/octo.git
python3.9 -m pip install -U setuptools wheel pip
python3.9 -m pip install -e octo
python3.9 -m pip install -r octo/requirements.txt
python3.9 -m pip install jax[cuda11_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3.9 -m pip install scipy==1.12.0 omegaconf
