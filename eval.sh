device=$1;
export CUDA_VISIBLE_DEVICES=$device;

# real data, fine-tuning on task1 then fine-tuning on real
python evaluation.py --data_name real --task_dir real --pretrain_dir task1
# real data, fine-tuning on real
python evaluation.py --data_name real --task_dir real
# real data, zero-shot
python evaluation.py --data_name real --task_dir real  --zero_shot
