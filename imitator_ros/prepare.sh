#!/usr/bin/bash

git clone https://github.com/ojh6404/imitator.git ~/imitator
git clone https://github.com/ojh6404/octo.git ~/octo
python3.11 -m pip install -e ~/imitator[cuda]
python3.11 -m pip install -e ~/octo
python3.11 -m pip install -U "jax[cuda12]"
