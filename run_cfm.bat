python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rat_rabbit.pth --source_species rat --target_species rabbit --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rat_rabbit.pth --target_species rabbit --src_species rat

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rat_pig.pth --source_species rat --target_species pig --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rat_pig.pth --target_species pig --src_species rat

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rat_mouse.pth --source_species rat --target_species mouse --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rat_mouse.pth --target_species mouse --src_species rat

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rabbit_rat.pth --source_species rabbit --target_species rat --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rabbit_rat.pth --target_species rat --src_species rabbit

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rabbit_pig.pth --source_species rabbit --target_species pig --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rabbit_pig.pth --target_species pig --src_species rabbit

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_rabbit_mouse.pth --source_species rabbit --target_species mouse --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_rabbit_mouse.pth --target_species mouse --src_species rabbit

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_pig_rat.pth --source_species pig --target_species rat --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_pig_rat.pth --target_species rat --src_species pig

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_pig_rabbit.pth --source_species pig --target_species rabbit --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_pig_rabbit.pth --target_species rabbit --src_species pig

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_pig_mouse.pth --source_species pig --target_species mouse --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_pig_mouse.pth --target_species mouse --src_species pig

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_mouse_rat.pth --source_species mouse --target_species rat --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_mouse_rat.pth --target_species rat --src_species mouse

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_mouse_rabbit.pth --source_species mouse --target_species rabbit --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_mouse_rabbit.pth --target_species rabbit --src_species mouse

python train_species.py --train_path ./species_data/train.pt --save_path ./model/cvae_mouse_pig.pth --source_species mouse --target_species pig --epoch 20 --lr 1e-4
python test_species.py --test_path ./species_data/test.pt --model_path ./model/cvae_mouse_pig.pth --target_species pig --src_species mouse
