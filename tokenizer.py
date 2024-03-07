# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
import argparse
from typing import List

from sentencepiece import SentencePieceProcessor

# 这里是指定的Llama SentencePiece分词器模型文件
TOKENIZER_MODEL = "tokenizer.model" 

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        # 初始化分词器，根据传入参数确定模型文件路径
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        # 加载SentencePieceProcessor模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        # 确保vocab_size与实际获取的pieces数量一致
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        # 对输入字符串进行编码，添加BOS/EOS标记（如果指定）
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        # 获取所有标记及其分数，并转换为浮点数形式
        tokens, scores = [], []
        for i in range(self.n_words):

            # 解码标记并进行轻量级后处理
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            # 对BOS和EOS标记进行特殊处理
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            # 将SentencePiece中的空白字符替换为空格
            t = t.replace('▁', ' ')
            # 将标记转换为UTF-8编码的字节形式
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)

        # 记录最长标记的长度
        max_token_length = max(len(t) for t in tokens)

        # 将数据写入二进制文件
        # tokenizer.bin 文件与 .model 文件内容相同，但扩展名为 .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            # 先写入最大标记长度
            f.write(struct.pack("I", max_token_length))
            # 遍历每个标记及其分数，将它们按照特定格式写入二进制文件
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)
    t.export()
