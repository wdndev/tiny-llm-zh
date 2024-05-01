from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "wdndev/tiny_llm_sft_92m"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

sys_text = "你是由wdndev开发的个人助手。"
# user_text = "中国的首都是哪儿？"
# user_text = "你叫什么名字？"
user_text = "介绍一下中国"
input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                        "<|user|>", user_text.strip(), 
                        "<|assistant|>"]).strip() + "\n"

model_inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=200)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


