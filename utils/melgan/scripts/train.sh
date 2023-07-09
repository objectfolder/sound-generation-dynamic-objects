python scripts/train_melgan.py \
    --save_path ./logs/`date +"%Y-%m-%dT%H-%M-%S"` \
    --data_path /viscam/projects/objectfolder_benchmark/benchmarks/Video_Sound_Prediction/DATA/features/melspec_10s_22050hz \
    --batch_size 64 --lr 1e-5