#!/bin/bash

python train.py --epochs 100 --backbone gin
python train.py --epochs 100 --backbone gcn -c
python train.py --epochs 100 --backbone gcn -s