
import os
# 设置环境变量，指定protobuf的Python实现为纯Python版本
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import json

def merge_tokenizer(llama_tokenizer_dir, chinese_sp_model_file, output_hf_dir="tinyllm_tokenizer_hf"):
    # 加载LlamaTokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    # 中文sentencepiece模型
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    # 将LlamaTokenizer加载为protobuf模型对象
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    # 将中文模型加载为protobuf模型对象
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # 打印基本信息
    print("llama token nums: ", len(llama_tokenizer))
    print("chinese sp nums: ", len(chinese_sp_model))

    # 向 LLaMA 的 tokenizer 中添加中文 tokens
    ## 1.首先创建一个set包含所有LLaMA的tokens以加速查找
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    ## 2.遍历中文模型的tokens，如果不在 LLaMA 的 tokens 集合中，则添加至 LLaMA 模型
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            # 创建新的SentencePiece对象
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece     # 设置token内容
            new_p.score = 0         # 设置默认的分数
            llama_spm.pieces.append(new_p)  # 添加到LLaMA的模型pieces中


    # 保存合并后的模型
    output_sp_dir = 'tmp_tinyllm_tokenizer_sp'  # 保存sentencepiece模型的目录
    os.makedirs(output_sp_dir, exist_ok=True)  # 确保目录存在
    # 保存sentencepiece模型到文件
    with open(output_sp_dir + '/tokenizer.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    
    # 使用新生成的vocab文件初始化LlamaTokenizer，并保存为 Hugging Face 格式
    tokenizer = LlamaTokenizer(vocab_file = output_sp_dir + '/tokenizer.model', legacy=True)
    ## 添加特殊 token
    custom_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|im_start|>", "<|im_end|>"]
    for token in custom_special_tokens:
        tokenizer.add_tokens(token)
    
    # vocab_dict = tokenizer.get_vocab()
    # with open('vocab_utf8.txt', 'w', encoding='utf-8') as f:
    #     json.dump(vocab_dict, f, indent=4)
    tokenizer.save_pretrained(output_hf_dir)
    print(f"tinyllm token num: {len(tokenizer)}")
    print(f"Tiny LLM tokenizer has been saved to {output_hf_dir}")

def test_tokenizer(hf_tokenizer_dir):
    tinyllm_tokenizer = LlamaTokenizer.from_pretrained(hf_tokenizer_dir)
    print("tinyllm tokenizer nums: ", len(tinyllm_tokenizer))

    sys_text = "你是由wdndev开发的个人助手。"
    user_text = "翻译下面的句子为英文：有朋自远方来，不亦乐乎"
    answer_text = "It is always a pleasure to greet a friend from afar."
    input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                            "<|user|>", user_text.strip(), 
                            "<|assistant|>"]).strip() + "\n" + answer_text.strip()

    print("-----input text: \n", input_txt)

    encode_ids = tinyllm_tokenizer.encode(input_txt, add_special_tokens=False)
    print("-----encode ids: \n", encode_ids)

    decode_ids = tinyllm_tokenizer.decode(encode_ids)
    print("-----dencode ids: \n", decode_ids)


if __name__ == "__main__":

    llama_tokenizer_dir = "input_dir/llama2_tokenizer"
    chinese_sp_model_file = "sp_output/chinese_spm_20000.model"
    output_hf_dir = "tinyllm_tokenizer_hf"

    # merge_tokenizer(llama_tokenizer_dir, chinese_sp_model_file, output_hf_dir)

    test_tokenizer(output_hf_dir)