#!/bin/bash

cd /home/hbui/Downloads/wheels
pip install --no-index --find-links . torch-1.13.1+cu116-cp310-cp310-linux_x86_64.whl
pip install --no-index --find-links . torchvision-0.14.1+cu116-cp310-cp310-linux_x86_64.whl


