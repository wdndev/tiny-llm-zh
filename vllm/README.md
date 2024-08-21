# Tiny LLM vLLM 模型部署

## 1.vLLM 环境

注意：测试环境为 vllm=0.4.0 

如果使用**CUDA 12 以上和PyTorch 2.1 以上**，可以直接使用以下命令安装vLLM。

```shell
pip install vllm==0.4.0
```

否则请参考vLLM官方的[安装说明](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

安装完成后，还需要以下操作~

1. 把 `vllm/tinyllm.py` 文件复制到env环境对应的 `vllm/model_executor/models` 目录下。
2. 然后在 `vllm/model_executor/models/__init__.py` 文件增加一行代码

```shell
"TinyllmForCausalLM": ("tinyllm", "TinyllmForCausalLM"),
```

> 由于模型结构是自己定义的，vllm官方未实现，需要自己手动加入

## 2.vLLM OpenAI API 接口

vLLM 部署实现 OpenAI API 协议的服务器非常方便。默认会在 http://localhost:8000 启动服务器。服务器当前一次托管一个模型，并实现列表模型、completions 和 chat completions 端口。

- completions：是基本的文本生成任务，模型会在给定的提示后生成一段文本。这种类型的任务通常用于生成文章、故事、邮件等。
- chat completions：是面向对话的任务，模型需要理解和生成对话。这种类型的任务通常用于构建聊天机器人或者对话系统。

在创建服务器时，可以指定模型名称、模型路径、聊天模板等参数。

- --host 和 --port 参数指定地址。
- --model 参数指定模型名称。
- --chat-template 参数指定聊天模板。
- --served-model-name 指定服务模型的名称。
- --max-model-len 指定模型的最大长度。

#### 启动服务

```shell
python -m vllm.entrypoints.openai.api_server \
    --served-model-name tinyllm_92m \
    --model wdn/tiny_llm_sft_92m \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
```
#### 查看当前模型列表

```shell
curl http://localhost:8000/v1/models
```

得到的返回值如下所示

```json
{
  "object": "list",
  "data": [
    {
      "id": "tinyllm_92m",
      "object": "model",
      "created": 1717735884,
      "owned_by": "vllm",
      "root": "tiny_llm_sft_92m",
      "parent": null,
      "permission": [
        {
          "id": "cmpl-55520539697749e7bc6f0243bf2dae18",
          "object": "model_permission",
          "created": 1720594920,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```
#### 测试OpenAI Completions API

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "tinyllm_92m",
        "prompt": "你好",
        "max_tokens": 50,
        "temperature": 0
    }'
```

得到返回值

```json
{
  "id": "cmpl-55520539697749e7bc6f0243bf2dae18",
  "object": "text_completion",
  "created": 1720594920,
  "model": "tinyllm_92m",
  "choices": [
    {
      "index": 0,
      "text": "你好，我是TinyLLM，一个由wdndev开发的人工智能助手。我可以回答各种问题、提供信息、执行任务和提供帮助。",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "total_tokens": 51,
    "completion_tokens": 50
  }
}
```

#### 使用Python脚本请求 OpenAI Completions API

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="tinyllm_92m",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(completion.choices[0].message)
```

返回值

```shell
ChatCompletionMessage(content='
你好，我是TinyLLM，一个由wdndev开发的人工智能助手。我可以回答各种问题、提供信息、执行任务和提供帮助。', role='assistant', function_call=None, tool_calls=None)
```

#### 使用curl测试 OpenAI Chat Completions API

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "tinyllm_92m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请介绍一下北京"}
        ]
    }'

```
返回结果
```json
{
    "id": "cmpl-55520539697749e7bc6f0243bf2dae18",
    "object": "chat.completion",
    "created": 1720594920,
    "model": "tinyllm_92m",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "：北京是中国的首都，也是中国改革开放的前沿城市之一，也是中国的首都。首都有着丰富的历史和文化底蕴，是中国的重要首都之一。"
            },
            "logprobs": null,
            "finish_reason": "stop",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 24,
        "total_tokens": 55,
        "completion_tokens": 31
    }
}
```

#### 使用 python 测试OpenAI Chat Completions API

```python
# vllm_openai_chat_completions.py
from openai import OpenAI
openai_api_key = "sk-xxx" # 随便填写，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="tinyllm_92m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ]
)
print(chat_outputs)
```

## 3.vLLM python调用

首先从 vLLM 库中导入 LLM 和 SamplingParams 类。LLM 类是使用 vLLM 引擎运行离线推理的主要类。SamplingParams 类指定采样过程的参数，用于控制和调整生成文本的随机性和多样性。

vLLM 提供了非常方便的封装，直接传入模型名称或模型路径即可，不必手动初始化模型和分词器。

```python
# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope。不设置的话，会从 huggingface 下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='/personal/wdn/tiny_llm_sft_92m' # 指定模型路径
    # model="wdn/tiny_llm_sft_92m" # 指定模型名称，自动下载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) 
    
    text = ["你好。",
            "请介绍一下北京。"]

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1, max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


