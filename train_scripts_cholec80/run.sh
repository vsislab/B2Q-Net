export CUDA_VISIBLE_DEVICES=0

# Training
python3 train.py phase --split cuhk4040 --bn_off --backbone convnextv2 --workers 4 --freeze --seq_len 256 --lr 1e-4 --random_seed --trial_name TRIAL_NAME --cfg configs/cholec80.yaml

# Evaluation
python3 save_predictions.py phase --split cuhk4040 --backbone convnextv2 --seq_len 256 --resume ../output/checkpoints/phase/[TRIAL_NAME]/models/checkpoint_best_acc.pth.tar --cfg configs/cholec80.yaml
