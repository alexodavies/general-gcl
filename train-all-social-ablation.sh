#!/bin/bash

python train.py --epochs 100 --backbone gin -c --exclude facebook_large
python train.py --epochs 100 --backbone gin -c --exclude twitch_egos
python train.py --epochs 100 --backbone gin -c --exclude cora
#python train.py --epochs 100 --backbone gin -c --exclude roads
#python train.py --epochs 100 --backbone gin -c --exclude fruit_fly
