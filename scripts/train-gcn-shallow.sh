#!/bin/bash

python train.py --epochs 100 --backbone gcn --num_gc_layers 3
python train.py --epochs 100 --backbone gcn --num_gc_layers 3 -c
python train.py --epochs 100 --backbone gcn --num_gc_layers 3 -s