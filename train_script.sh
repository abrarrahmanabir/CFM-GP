#!/bin/bash



python train.py --train_path ./data/train_covid_gene.pt --val_path ./data/valid_covid_gene.pt --save_path ./model/covid.pt --epoch 50 --lr 1e-4


python train.py --train_path ./data/train_pbmc_gene.pt --val_path ./data/valid_pbmc_gene.pt --save_path ./model/pbmc.pt --epoch 50 --lr 1e-4


python train.py --train_path ./data/lupus_train_gene.pt --val_path ./data/lupus_valid_gene.pt --save_path ./model/lupus.pt --epoch 50 --lr 1e-4


python train.py --train_path ./data/statefate_train_gene.pt --val_path ./data/statefate_valid_gene.pt --save_path ./model/statefate.pt --epoch 50 --lr 1e-4


python train.py --train_path ./data/glio_train.pt --val_path ./data/glio_valid.pt --save_path ./model/glio.pt --epoch 50 --lr 1e-4


