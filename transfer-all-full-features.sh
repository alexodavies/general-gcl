#!/bin/bash

for checkpoint_name in  untrained chem-100.pt social-100.pt all-100.pt
do
    python features_transfer.py --checkpoint "$checkpoint_name" -f --num 5000000 --backbone gin --batch_size 512
done
#
#for checkpoint_name in  gat-all.pt gat-chem.pt gat-social.pt
#do
#    python features_transfer.py --checkpoint "$checkpoint_name" -f --num 5000000 --backbone gat
#done
#
#
#for checkpoint_name in  untrained gcn-all.pt gcn-chem.pt gcn-social.pt
#do
#    python features_transfer.py --checkpoint "$checkpoint_name" -f --num 5000000 --backbone gcn
#done
