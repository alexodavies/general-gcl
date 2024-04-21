#!/bin/bash

python train.py --epochs 100 --backbone gin
python train.py --epochs 100 --backbone gin -c
python train.py --epochs 100 --backbone gin -s