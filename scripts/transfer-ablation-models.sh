#!/bin/bash

for checkpoint_name in cora-excluded.pt facebook-excluded.pt fly-excluded.pt twitch-excluded.pt twitch-excluded.pt
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
