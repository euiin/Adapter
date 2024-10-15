export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=6 python run_glue.py \
--model_name_or_path roberta-base \
--lora_path checkpoint/roberta_base_lora_rte.bin \
--task_name rte \
--do_predict \
--output_dir ./results/test/roberta-base_lora_rte \
--apply_lora \
--lora_r 16 \
--lora_alpha 32
# torch.distributed.launch --nproc_per_node=1