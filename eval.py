from peft import PeftModel
from transformers import pipeline

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./mentalchat_qwen_finetuned")

mental_health_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)


def generate_response(user_input):
    system_message = "你是一位富有同理心的心理健康助手。请根据用户的倾诉，提供温暖、支持性的回应。"
    
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    outputs = mental_health_pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    return outputs[0]['generated_text']

test_queries = [
    "我最近感到压力很大，晚上睡不着觉。",
    "我觉得很孤独，没有人理解我。",
    "我对任何事情都提不起兴趣，我是不是抑郁了？"
]

for query in test_queries:
    print(f"用户: {query}")
    response = generate_response(query)
    print(f"助手: {response}\n")