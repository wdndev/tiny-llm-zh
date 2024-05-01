from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "wdndev/tiny_llm_ptm_92m"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

# text = "人生如棋，落子无悔，"
# text = "《小王子》是一本畅销童话书，它讲述了："
text = "床前明月光，疑是地上霜。举头望明月，"

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=200)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

