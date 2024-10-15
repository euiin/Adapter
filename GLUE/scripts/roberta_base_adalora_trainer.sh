CUDA_VISIBLE_DEVICES=4 python run_glue_adalora_peft.py \
--model_name_or_path roberta-base \
--task_name rte \
--apply_adalora \
--apply_lora \
--lora_type svd \
--target_rank 1  --lora_r 3  \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 50000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--target_modules 'query', 'key', 'value', 'intermediate.dense', 'output.dense' \
--lora_alpha 16 \
--do_train \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 5e-4 --num_train_epochs 7 \
--warmup_steps 1000 \
--weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 6 \
--output_dir ./results/roberta-base_adalora_rte \
--overwrite_output_dir \
--wandb_project "Meta-Lora" \
--run_name "roberta-base_rte_adalora" \
--wandb_watch "all" \
--wandb_log_model "checkpoint" \


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