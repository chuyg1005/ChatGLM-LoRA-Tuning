# 生成pretrain-data
python code/process_re.py --data_dir ./xxx --save_dir ./xxx


# train

conda activate chuyg-glm
unset CUDA_VISIBLE_DEVICES
## from scratch: pretrain
deepspeed --include=localhost:1 --master_port=29500 train_deepspeed.py --data_dir en
## from pretrain: finetune
deepspeed --include=localhost:1 --master_port=29500 train_deepspeed.py --data_dir en --pretrain_path=./checkpoint/xxx


# evaluate
export CUDA_VISIBLE_DEVICES=0
## 随机初始化[用不到]
python evaluation.py --data_dir en --from_scratch

## 使用下面的
python evaluation.py --data_dir en
