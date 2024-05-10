#!/usr/bin/bash

sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U setuptools wheel pip
python3.9 -m pip install -r requirements.txt
