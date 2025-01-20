export CUDA_VISIBLE_DEVICES=0

python3 save_predictions.py phase --split cuhk4040 --backbone convnextv2 --seq_len 256 --resume ../output/checkpoints/phase/[TRIAL_NAME]/models/checkpoint_best_acc.pth.tar --cfg configs/cholec80.yaml
