from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen2.5-7B"
adapter_path = "./mentalchat_qwen_finetuned_stable"

# 使用与训练相同的量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载量化后的基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, adapter_path)

def generate_response(user_input):
    system_message = "你是一位富有同理心的心理健康助手。请根据用户的倾诉，提供温暖、支持性的回应。"
    
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # 使用模型的 generate 方法而不是 pipeline
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # 解码输出，跳过输入部分
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除输入部分，只返回生成的回复
    return response[len(prompt):]

# 测试
test_queries = [
    "我最近感到压力很大，晚上睡不着觉。",
    "我觉得很孤独，没有人理解我。",
    "我对任何事情都提不起兴趣，我是不是抑郁了？"
]

for query in test_queries:
    print(f"用户: {query}")
    response = generate_response(query)
    print(f"助手: {response}\n")