#!/bin/bash

for checkpoint_name in  untrained gat-all.pt gat-chem.pt gat-social.pt
do
    python features_transfer.py --checkpoint "$checkpoint_name" -f --num 5000000 --backbone gat
done
