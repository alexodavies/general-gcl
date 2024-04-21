#!/bin/bash



for checkpoint_name in  untrained all-100.pt chem-100.pt social-100.pt
do
    python node_classification_transfer.py --checkpoint "$checkpoint_name" -f --epochs 10 --backbone gin
done

for checkpoint_name in  untrained gat-all.pt gat-chem.pt gat-social.pt
do
    python node_classification_transfer.py --checkpoint "$checkpoint_name" -f --epochs 10 --backbone gat
done


for checkpoint_name in  untrained gcn-all-shallow.pt gcn-all.pt gcn-chem.pt gcn-social.pt
do
    python node_classification_transfer.py --checkpoint "$checkpoint_name" -f --epochs 10 --backbone gcn
done
