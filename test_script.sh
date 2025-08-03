#!/bin/bash



python test.py --test_path ./data/test_covid_gene.pt --model_path ./model/covid.pt --save_dir ./covid_results/ --out_prefix covid


python test.py --test_path ./data/test_pbmc_gene.pt --model_path ./model/pbmc.pt --save_dir ./pbmc_results/ --out_prefix pbmc


python test.py --test_path ./data/lupus_test_gene.pt --model_path ./model/lupus.pt --save_dir ./lupus_results/ --out_prefix lupus


python test.py --test_path ./data/statefate_test_gene.pt --model_path ./model/statefate.pt --save_dir ./statefate_results/ --out_prefix statefate


python test.py --test_path ./data/glio_test.pt --model_path ./model/glio.pt --save_dir ./glio_results/ --out_prefix glio


