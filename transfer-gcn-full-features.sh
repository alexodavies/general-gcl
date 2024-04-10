#!/bin/bash

for checkpoint_name in  untrained gcn-all.pt gcn-chem.pt gcn-social.pt
do
    python features_transfer.py --checkpoint "$checkpoint_name" -f --num 5000000 --backbone gcn
done
