from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from transformers import DataCollatorForLanguageModeling

# 1. 加载数据
dataset = load_dataset("ShenLab/MentalChat16K")
model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 数据预处理
def format_instruction(sample):
    system_message = "你是一位富有同理心的心理健康助手。请根据用户的倾诉，提供温暖、支持性的回应。"
    user_input = sample.get("input", "") or ""
    assistant_output = sample.get("output", "") or ""

    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_text}

# 应用格式化
tokenized_dataset = dataset.map(format_instruction)

# 3. 检查是否有验证集
if "validation" not in tokenized_dataset:
    train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
    tokenized_dataset["train"] = train_val_split["train"]
    tokenized_dataset["validation"] = train_val_split["test"]

print(f"训练集大小: {len(tokenized_dataset['train'])}")
print(f"验证集大小: {len(tokenized_dataset['validation'])}")

# 4. 模型配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 改为 True 可能更稳定
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    use_cache=True,
)

# 关键步骤：准备模型用于k-bit训练
model = prepare_model_for_kbit_training(model)

# 5. LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # 提取文本
        texts = [feature["text"] for feature in features]
        
        # 使用tokenizer进行批处理，启用动态填充
        batch = self.tokenizer(
            texts,
            padding=True,  # 动态填充
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        
        # 设置labels
        batch["labels"] = batch["input_ids"].clone()
        
        return batch

data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    mlm=False,
)


# 6. 训练配置
training_args = TrainingArguments(
    output_dir="./mentalchat_qwen_finetuned_stable",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    
    # 验证配置
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # 梯度稳定化配置
    max_grad_norm=0.5,
    gradient_checkpointing=False,
    
    # 优化器配置
    optim="paged_adamw_8bit",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    # 学习率调度
    lr_scheduler_type="cosine",
    
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    fp16=True,
)

# 7. 使用 SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,

    data_collator=data_collator,  # 使用自定义数据整理器
)

print("开始训练...")

# 确保模型处于训练模式
model.train()


trainer.train()
trainer.save_model()
print("训练完成！")