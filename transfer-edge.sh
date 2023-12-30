#!/bin/bash

for checkpoint_name in edge-views-chem.pt edge-views-social.pt
do
    python transfer.py --checkpoint "$checkpoint_name"
done
