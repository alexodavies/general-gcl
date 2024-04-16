#!/bin/bash


#python node_classification_transfer.py --checkpoint untrained --epochs 10 --backbone gin -f


for checkpoint_name in  untrained all-100.pt chem-100.pt social-100.pt edge-views-all.pt
do
    python node_classification_transfer.py --checkpoint "$checkpoint_name" --epochs 5 --backbone gin  --batch_size 512 -f
done


for checkpoint_name in  untrained all-100.pt chem-100.pt social-100.pt edge-views-all.pt
do
    python node_classification_transfer.py --checkpoint "$checkpoint_name" --epochs 5 --backbone gin  --batch_size 512
done

#for checkpoint_name in  untrained all-100.pt chem-100.pt social-100.pt edge-views-all.pt
#do
#python node_classification_transfer.py --checkpoint edge-views-all.pt --epochs 4 --backbone gin -f  --batch_size 512
#done

#for checkpoint_name in  untrained gat-all.pt gat-chem.pt gat-social.pt
#do
#    python node_classification_transfer.py --checkpoint "$checkpoint_name" --epochs 10 --backbone gat
#done
#
#
#for checkpoint_name in  untrained gcn-all-shallow.pt gcn-all.pt gcn-chem.pt gcn-social.pt
#do
#    python node_classification_transfer.py --checkpoint "$checkpoint_name" --epochs 10 --backbone gcn
#done
