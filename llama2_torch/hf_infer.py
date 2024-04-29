from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_id = "wdndev/hf_tiny_llm_58m_sft"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

# text = "介绍一下刘德华。"
# text = "请问，世界上最大的动物是什么？"
text = "中国的首都在什么地方？"
# text = "请用C++语言实现一个冒泡排序算法。"

start_time = time.time()

# 哎。。。，SFT时没有注意这个特殊的token，拼接了prompt和answer，使用HF时，词表中没有这个，，，难受
model_inputs_id = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.special_tokens['<bos>']]
model_inputs_id = (torch.tensor(model_inputs_id, dtype=torch.long, device=model.device)[None, ...])
generated_ids = model.generate(model_inputs_id)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_id, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(time.time() - start_time)

print(response)