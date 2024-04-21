#!/bin/bash

for checkpoint_name in all-100.pt chem-100.pt social-100.pt
do
    python transfer.py --checkpoint "$checkpoint_name" -f --num 5000000
done
