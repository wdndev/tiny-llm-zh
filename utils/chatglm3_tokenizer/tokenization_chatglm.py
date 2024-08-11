import json
import os
import re
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


logger = logging.get_logger(__name__)


class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        # 确保vocab_size与piece数量一致
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # 定义聊天角色相关的特殊token
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        # 添加额外的通用特殊token
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        # 创建特殊token与ID之间的映射关系
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            # 分配新的词汇表ID给特殊token
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        # 生成正则表达式，用于在apply_chat_template方法中查找特殊token
        self.role_special_token_expression = "|".join([re.escape(token) for token in special_tokens]) # for apply_chat_template

    def tokenize(self, s: str, encode_special_tokens=False):
        """ 对输入字符串进行分词操作，可选择是否编码特殊token
        """
        if encode_special_tokens:
            # 对特殊字符进行处理
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                # 查找并保留非特殊token部分的分词结果
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                # 直接添加特殊token
                t.append(s[match.start():match.end()])
                last_index = match.end()
            # 处理剩余非特殊token部分
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            # 当encode_special_tokens为False时，直接调用SentencePiece模型进行分词
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """ 将字符串转化为ID列表，可选择是否添加BOS/EOS token
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """ 将ID列表解码为字符串
        """
        text, buffer = "", []
        for token in t:
            # 处理特殊tokenID转字符串
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        # 解码剩余普通tokenID
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        """ 将分词结果（List[str]）解码为字符串
        """
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ 将给定的token字符串转化为对应的ID
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """ 将给定的ID转化为对应的token字符串
        """
        # 处理特殊tokenID
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        # 处理边界情况和其他特殊ID
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        # 将普通ID转换为token
        return self.sp_model.IdToPiece(index)


class ChatGLMTokenizer(PreTrainedTokenizer):
    # 预训练模型所需的文件名配置，这里指向tokenizer的model文件
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    # 模型输入的特征名称列表
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
        self,
        vocab_file,
        padding_side="left",
        clean_up_tokenization_spaces=False,
        encode_special_tokens=False,
        **kwargs
    ):
        # 设置tokenizer的名称
        self.name = "GLMTokenizer"
        # 存储vocab文件路径
        self.vocab_file = vocab_file
        # 使用SPTokenizer作为基础分词器
        self.tokenizer = SPTokenizer(vocab_file)
        # 定义特殊token及其对应的ID
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<unk>": self.tokenizer.pad_id,
            "<pad>": self.tokenizer.pad_id
        }
        self.encode_special_tokens = encode_special_tokens

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

        # self.chat_template = "{% for message in messages %}{% if loop.first %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"

    def get_command(self, token):
        """ 获取指定特殊 token 对应的 id
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        # 如果不在自定义特殊 token 中，则从基础SPTokenizer的特殊 token 中查找
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        """ 通过ID获取未登录词、填充符和结束符的字符串形式
        """
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<unk>"))

    @property
    def pad_token(self) -> str:
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<pad>"))

    @property
    def eos_token(self) -> str:
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<eos>"))

    @property
    def unk_token_id(self) -> int:
        """ 获取未登录词、填充符和结束符的ID形式
        """
        return self.get_command("<unk>")

    @property
    def pad_token_id(self) -> int:
        return self.get_command("<pad>")

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @unk_token.setter
    def unk_token(self, value):
        """ 不支持设置未登录词、填充符和结束符，输出警告信息
        """
        logger.warning("Setting unk_token is not supported, use the default one.")

    @pad_token.setter
    def pad_token(self, value):
        logger.warning("Setting pad_token is not supported, use the default one.")

    @eos_token.setter
    def eos_token(self, value):
        logger.warning("Setting eos_token is not supported, use the default one.")

    @property
    def vocab_size(self):
        """ 返回整个词汇表的大小
        """
        return self.tokenizer.n_words

    def get_vocab(self):
        """ 获取词汇表字典，其中键是token，值是其对应的ID
        """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        """ 实现分词功能，利用SPTokenizer进行分词操作
        """
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):
        """ 将token字符串转化为ID
        """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """ 将ID转化为token字符串
        """
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ 将分词结果的tokens列表还原为字符串
        """
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """ 将词汇表和特殊令牌token保存到指定目录。

        Args:
            save_directory (`str`): 将词汇表和特殊令牌文件保存到指定目录。
            filename_prefix (`str`, *optional*): 可选添加到保存文件名前的前缀。

        Returns:
            `Tuple(str)`: 保存文件的路径
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        """ 获取用于模型输入的前缀 token 
        """
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message):
        """ 构建单条消息的 token 序列
        """
        assert role in ["system", "user", "assistant", "observation"], role
        # 构建角色标识Token序列
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        # 构建消息正文Token序列
        message_tokens = self.tokenizer.encode(message)
        # 合并角色标识Token与消息正文Token
        tokens = role_tokens + message_tokens
        return tokens

    def build_chat_input(self, query, history=None, role="user"):
        """ 根据对话历史及当前query构建模型输入
        """
        if history is None:
            history = []
        input_ids = []
        # 遍历对话历史
        for item in history:
            # 获取内容
            content = item["content"]
            # 若为系统消息且包含工具信息，将其加入内容
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            # 构建单条历史消息的Token序列并加入到模型输入ID列表
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        # 构建当前query的Token序列并加入到模型输入ID列表
        input_ids.extend(self.build_single_message(role, "", query))
        # 添加表示回复的assistant标记
        input_ids.extend([self.get_command("<|assistant|>")])
        # 调用tokenizer批量编码方法，返回PyTorch张量形式的模型输入
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """ 通过拼接和添加特殊标记，从一个或两个序列构建用于序列分类任务的模型输入。
        
        BERT序列格式如下：
        - 单一序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`): 将添加特殊token的IDs列表
            token_ids_1 (`List[int]`, *optional*): 可选的第二个序列的IDs列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的[输入IDs](../glossary#input-ids)列表。
        """
        # 获取前缀标记
        prefix_tokens = self.get_prefix_tokens()
        # 在token_ids_0前添加前缀标记
        token_ids_0 = prefix_tokens + token_ids_0
        # 若存在token_ids_1，将token_ids_0、token_ids_1连接，并添加结束标记，然后返回
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """ 此方法用于对编码后的输入进行填充（左右两侧填充，直至达到预设长度或批次中的最大长度）

        Args:
            encoded_inputs: 字典形式的编码后输入，键为特征名称，值为整数列表（例如，`List[int]`），或者一批编码后的输入（例如，`List[List[int]]`）。
            max_length: 返回列表的最大长度，也可作为填充长度
            padding_strategy: 填充策略，有以下选项：
                - PaddingStrategy.LONGEST : 根据批次中最长序列进行填充
                - PaddingStrategy.MAX_LENGTH: 默认策略，填充至最大长度
                - PaddingStrategy.DO_NOT_PAD: 不进行填充
                本tokenizer的填充方向由self.padding_side属性决定：
                    - 'left': 在序列左侧填充
                    - 'right': 在序列右侧填充
            pad_to_multiple_of: （可选）若设置，则将序列填充至给定值的倍数。这对于在NVIDIA硬件上启用具有计算能力`>= 7.5`（Volta及以上）的Tensor Core非常有用。
            return_attention_mask:（可选）若设置为False，则避免返回注意力掩码（默认：根据模型特性设置
        """
        # 从模型默认设置中加载填充侧信息
        assert self.padding_side == "left"

        # 获取必要的输入特征，这里假设第一个特征为主要输入特征
        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        # 如果填充策略为最长序列，则将最大长度设置为当前序列长度
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # 计算实际最大长度，确保满足pad_to_multiple_of的要求
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 判断是否需要填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 若不存在注意力掩码，则初始化
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        # 若需要填充，则执行填充操作
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            # 对注意力掩码进行填充
            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            # 对位置标识进行填充
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            # 对主要输入特征进行填充
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
