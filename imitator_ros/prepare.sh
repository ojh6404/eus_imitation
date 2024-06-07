#!/usr/bin/bash

git clone https://github.com/ojh6404/imitator.git ~/imitator
git clone https://github.com/ojh6404/octo.git ~/octo
python3.11 -m pip install -e ~/imitator[cuda]
python3.11 -m pip install -e ~/octo
# python3.11 -m pip install --upgrade "jax[cuda11_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3.11 -m pip install -U "jax[cuda12]"
