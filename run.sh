#!/bin/bash

set -e  # 如果某条命令失败，立即退出

# 训练 ChatGLM3-6B
echo "开始训练 ChatGLM3-6B..."
nohup python main.py --base "THUDM/chatglm3-6b" --output_dir "./model/finetuned-chatglm3-6B" > ./logs/train/chatglm3-6B-train.log 2>&1

# 训练 Qwen2.5-7B
echo "开始训练 Qwen2.5-7B..."
nohup python main.py --base "Qwen/Qwen2.5-7B" --output_dir "./model/finetuned-Qwen2.5-7B" > ./logs/train/Qwen2.5-7B-train.log 2>&1

# 训练 Qwen1.5-4B
echo "开始训练 Qwen1.5-4B..."
nohup python main.py --base "Qwen/Qwen1.5-4B" --output_dir "./model/finetuned-Qwen1.5-4B" > ./logs/train/Qwen1.5-4B-train.log 2>&1

# 运行 scorer.py
echo "开始运行 scorer.py..."
nohup python scorer.py \
  --base_model "Qwen/Qwen1.5-4B" \
  --adapter_path "./model/finetuned-Qwen1.5-4B" \
  --eval_chatglm true \
  --chatglm_adapter_path "./model/finetuned-chatglm3-6B" \
  --eval_qwen2_5 true \
  --qwen2_5_adapter_path "./model/finetuned-Qwen2.5-7B" \
  --num_samples 30 \
  > ./logs/scorer/scorer.log 2>&1

echo "开始运行 plot.py..."
# 对ChatGLM3-6B的训练过程绘图
python ./plot.py \
 --log_file "./logs/train/chatglm3-6B-train.log" \
 --output_dir "./figures/ChatGLM3-6B/" \
 --dpi 300 \
 --model_name "ChatGLM3-6B"\
 --detailed \
 --export_stats

# 对Qwen1.5-4B的训练过程绘图
python ./plot.py \
 --log_file "./logs/train/Qwen1.5-4B-train.log" \
 --output_dir "./figures/Qwen1.5-4B/" \
 --dpi 300 \
 --model_name "Qwen1.5-4B"\
 --detailed \
 --export_stats

# 对Qwen2.5-7B的训练过程绘图
python ./plot.py \
 --log_file "./logs/train/Qwen2.5-7B-train.log" \
 --output_dir "./figures/Qwen2.5-7B/" \
 --dpi 300 \
 --model_name "Qwen2.5-7B"\
 --detailed \
 --export_stats

echo "全部任务完成！"