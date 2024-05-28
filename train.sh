device=$1;
# real data, fine-tuning on tasks1
deepspeed --include=localhost:$device train_deepspeed.py --task_dir=real --data_name=real
# real data, fine-tuning on task1 then fine-tuning on real
deepspeed --include=localhost:$device train_deepspeed.py --task_dir=real --data_name=real --pretrain_dir=task1
