from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
import torch
from transformers import DataCollatorForLanguageModeling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default="Qwen/Qwen1.5-4B")
parser.add_argument('--output_dir', type=str, default="./mentalchat_finetuned")
parser.add_argument('--lora_r', type=int, default=16, help='LoRA秩')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA缩放参数')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
args = parser.parse_args()

# 1. 加载数据
print("加载数据集...")
dataset = load_dataset("ShenLab/MentalChat16K")
model_name = args.base

# 2. 加载tokenizer
print(f"加载tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 数据预处理
def format_instruction(sample):
    """格式化对话数据"""
    system_message = "You are an empathetic mental health assistant. Please provide warm and supportive responses based on the user's story."
    user_input = sample.get("input", "") or ""
    assistant_output = sample.get("output", "") or ""

    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output}
    ]
    
    # 尝试使用chat_template，如果不支持则使用简单格式
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # 对于不支持chat_template的模型，使用简单格式
        formatted_text = f"System: {system_message}\nUser: {user_input}\nAssistant: {assistant_output}"
    
    return {"text": formatted_text}

# 应用格式化
print("格式化数据...")
tokenized_dataset = dataset.map(format_instruction)

# 4. 检查是否有验证集
if "validation" not in tokenized_dataset:
    print("划分训练集和验证集...")
    train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
    tokenized_dataset["train"] = train_val_split["train"]
    tokenized_dataset["validation"] = train_val_split["test"]

print(f"训练集大小: {len(tokenized_dataset['train'])}")
print(f"验证集大小: {len(tokenized_dataset['validation'])}")

# 5. 模型配置
print(f"加载模型: {model_name}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    use_cache=False,
)

# 关键步骤：准备模型用于k-bit训练
model = prepare_model_for_kbit_training(model)

# 6. LoRA配置 - 根据模型类型动态设置target_modules
def get_target_modules(model_name, model):
    """根据模型类型获取LoRA目标模块"""
    model_name_lower = model_name.lower()
    
    print("获取模型模块信息...")
    module_names = [name for name, _ in model.named_modules()]
    
    if "chatglm" in model_name_lower:
        # ChatGLM模型的模块结构
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        print(f"ChatGLM模型检测到，使用target_modules: {target_modules}")
    elif "qwen" in model_name_lower:
        # Qwen模型的模块结构
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"Qwen模型检测到，使用target_modules: {target_modules}")
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        # LLaMA/Mistral模型的模块结构
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"LLaMA/Mistral模型检测到，使用target_modules: {target_modules}")
    elif "phi" in model_name_lower:
        # Phi模型的模块结构
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
        print(f"Phi模型检测到，使用target_modules: {target_modules}")
    else:
        # 通用设置，尝试常见的模块名称
        target_modules = []
        possible_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
                          "query_key_value", "dense", "fc1", "fc2", "W_pack"]
        
        for module in possible_modules:
            if any(module in name for name in module_names):
                target_modules.append(module)
        
        if not target_modules:
            print(f"警告: 无法自动检测合适的target_modules，使用默认设置")
            print(f"可用的模块: {module_names[:20]}...")
            target_modules = ["q_proj", "k_proj", "v_proj"]
    
    valid_target_modules = []
    for module in target_modules:
        if any(module in name for name in module_names):
            valid_target_modules.append(module)
        else:
            print(f"警告: 模块 '{module}' 不在模型中")
    
    if not valid_target_modules:
        raise ValueError(f"没有找到有效的target_modules。请检查模型结构。可用模块: {module_names[:50]}...")
    
    print(f"最终使用的target_modules: {valid_target_modules}")
    return valid_target_modules

target_modules = get_target_modules(model_name, model)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. 数据整理器
class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        texts = [feature["text"] for feature in features]
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        
        batch["labels"] = batch["input_ids"].clone()
        
        return batch

data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    mlm=False,
)

# 8. 训练配置
print("设置训练参数...")
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1 if args.base == 'THUDM/chatglm3-6b' else 4,
    per_device_eval_batch_size=1 if args.base == 'THUDM/chatglm3-6b' else 4,
    gradient_accumulation_steps=4 if args.base == 'THUDM/chatglm3-6b' else 2,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=1000,
    
    # 验证配置
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # 梯度稳定化配置
    max_grad_norm=0.5,
    gradient_checkpointing=True,
    
    # 优化器配置
    optim="paged_adamw_8bit",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    
    # 学习率调度
    lr_scheduler_type="cosine",
    
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    report_to="none",
    torch_compile=False,  # 对于某些模型，torch_compile可能导致问题
    remove_unused_columns=False,
    fp16=True,
)

# 9. 使用 SFTTrainer
print("创建训练器...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True,
)

print("开始训练...")
model.train()

# 开始训练
trainer.train()

# 保存模型
print(f"保存模型到: {args.output_dir}")
trainer.save_model()
print("训练完成！")

# 保存训练配置
with open(f"{args.output_dir}/training_config.json", "w") as f:
    import json
    json.dump({
        "base_model": model_name,
        "target_modules": target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "output_dir": args.output_dir
    }, f, indent=2)