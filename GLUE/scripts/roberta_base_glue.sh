# #stsb
# export num_gpus=2
# #export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# #export PYTHONHASHSEED=0
# export output_dir="./results/roberta-base_lora_stsb"
# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
# examples/text-classification/run_glue.py \
# --model_name_or_path roberta-base \
# --lora_path ./roberta_base_mnli_lora.bin \
# --task_name stsb \
# --do_train \
# # --do_eval \
# --max_seq_length 512 \
# --per_device_train_batch_size 16 \
# --learning_rate 4e-4 \
# --num_train_epochs 40 \
# --output_dir $output_dir/model \
# --overwrite_output_dir \
# --logging_steps 10 \
# --logging_dir $output_dir/log \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --warmup_ratio 0.06 \
# --apply_lora \
# --lora_r 8 \
# --lora_alpha 16 \
# --seed 0 \
# --weight_decay 0.1

# #rte
# export output_dir="./results/roberta-base_lora_rte"
# python -m torch.distributed.launch --nproc_per_node=$num_gpus \
# examples/text-classification/run_glue.py \
# --model_name_or_path roberta-base \
# --lora_path ./roberta_base_mnli_lora.bin \
# --task_name rte \
# --do_train \
# --do_eval \
# --max_seq_length 512 \
# --per_device_train_batch_size 32 \
# --learning_rate 5e-4 \
# --num_train_epochs 80 \
# --output_dir $output_dir/model \
# --overwrite_output_dir \
# --logging_steps 10 \
# --logging_dir $output_dir/log \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --warmup_ratio 0.06 \
# --apply_lora \
# --lora_r 8 \
# --lora_alpha 16 \
# --seed 0 \
# --weight_decay 0.1

# #cola
# export output_dir="./results/roberta-base_lora_cola"
# python -m torch.distributed.launch --nproc_per_node=$num_gpus \
# examples/text-classification/run_glue.py \
# --model_name_or_path roberta-large \
# --task_name cola \
# --do_train \
# --do_eval \
# --max_seq_length 128 \
# --per_device_train_batch_size 4 \
# --learning_rate 3e-4 \
# --num_train_epochs 20 \
# --output_dir $output_dir/model \
# --logging_steps 10 \
# --logging_dir $output_dir/log \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --warmup_ratio 0.06 \
# --apply_lora \
# --lora_r 8 \
# --lora_alpha 16 \
# --seed 0 \
# --weight_decay 0.1

export num_gpus=4
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir='./roberta-large_rte_vera/model'
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
run_glue.py \
    --model_name_or_path roberta-large \
    --task_name rte \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-2 \
    --learning_rate_head 2e-3 \
    --num_train_epochs 40 \
    --output_dir './roberta-large_rte_vera/model' \
    --overwrite_output_dir \
    --logging_steps 10 \
    --save_steps 1000 \
    --logging_dir './roberta-large_rte_vera/log' \
    --evaluation_strategy no \
    --save_strategy steps \
    --warmup_ratio 0.06 \
    --apply_vera True \
    --lora_r 1024 \
    --seed 0 \
    --weight_decay 0.1 \
    --target_modules ["query", "value"] \
    --modules_to_save "classifier" \
    --wandb_project "Meta-Lora" \
    --run_name "roberta-large_rte_vera" \
    --wandb_watch "all" \
	--wandb_log_model "checkpoint" \