export CUDA_VISIBLE_DEVICES=0

python3 train.py phase --split cuhk4040 --bn_off --backbone convnextv2 --workers 4 --freeze --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --cfg configs/cholec80.yaml
