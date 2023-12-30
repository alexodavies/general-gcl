#!/bin/bash

for checkpoint_name in node-views-all.pt node-views-chem.pt node-views-social.pt
do
    python transfer.py --checkpoint "$checkpoint_name"
done
