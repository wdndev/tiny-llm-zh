import re
import torch
import numpy as np
from queue import Queue
from typing import Tuple, List, Union, Iterable
from transformers.utils import logging, add_start_docstrings
from transformers.generation.logits_process import LogitsProcessor, LOGITS_PROCESSOR_INPUTS_DOCSTRING, LogitsProcessorList

def make_context(model, tokenizer, 
                 messages: List[dict], 
                 system: str = "You are a helpful assistant.",
                 max_new_tokens: int=0, 
                ):
    # 确定新生成的token数量，优先使用传入参数，否则使用模型配置中的默认值
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    # 计算模型允许的最大输入长度（模型最大长度减去新生成的token数）
    max_input_length = model.config.max_position_embeddings - max_new_tokens

    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def _parse_messages(messages):
        """ 解析消息列表，分离系统消息、查询和对话历史
        """
        system, query, history = "", "", []
        ## system
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        ## query
        ### 确保最后一项是用户消息
        assert messages[-1]["role"] == "user"
        query = messages[-1]["content"]
        messages = messages[:-1]
        ## history
        assert len(messages) % 2 == 0
        for i in range(0, len(messages), 2):
            assert messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant"
            history.append([messages[i]["content"], messages[i+1]["content"]])

        return system, query, history
    
    # 调用_parse_messages解析消息
    _system, query, history = _parse_messages(messages)

    ## system
    system_text = _system if _system != "" else system
    system_tokens = []
    if system_text:
        # system_tokens = tokenizer.build_single_message("system", "", system_text.strip())
        system_tokens = tokenizer.encode(text=("<|system|>\n"+system_text.strip()), add_special_tokens=True, truncation=True) + nl_tokens
    ## query
    # query_tokens = tokenizer.build_single_message("user", "", query.strip())
    query_tokens = tokenizer.encode(text=("<|user|>\n"+query.strip()), add_special_tokens=False, truncation=True) + nl_tokens
    ## final assistant
    # final_tokens = tokenizer.build_single_message("assistant", "", "")
    final_tokens = tokenizer.encode("<|assistant|>", add_special_tokens=False, truncation=True) + nl_tokens
    
    ## max_history_tokens
    max_history_length = max_input_length - len(system_tokens) - len(query_tokens) - len(final_tokens)
    
    ## history
    ## 逆序遍历对话历史，构建token序列
    context_tokens = []
    for turn_query, turn_response in reversed(history):
        ## query tokens
        history_query_tokens = tokenizer.encode("<|user|>\n"+turn_query.strip(), add_special_tokens=False, truncation=True) + nl_tokens
        ## answer tokens
        histroy_response_tokens = tokenizer.encode("<|assistant|>\n"+turn_response.strip(), add_special_tokens=False, truncation=True) + nl_tokens
        ## this round tokens
        next_context_tokens = history_query_tokens + histroy_response_tokens
        ## concat
        ## 确保加入这些token后总长度不超过允许的最大历史长度
        current_context_size = len(next_context_tokens) + len(context_tokens)
        if current_context_size < max_history_length:
            context_tokens = next_context_tokens + context_tokens
        else:
            break
    input_tokens = system_tokens + context_tokens + query_tokens + final_tokens

    return torch.LongTensor([input_tokens]).to(model.device)

def parse_pot_no_stream(inputs):
    """ 解析并处理输入字符串中特定格式（形如 <<...>>）的代码片段。
        这些代码片段可以是简单的数学表达式赋值，也可以是定义和调用函数。
        1. 对于包含 "func" 的代码片段，它会识别函数定义，执行该函数，
           并将函数返回的结果替换到原始字符串中的相应位置。
           如果函数涉及到 sympy（一个符号计算库），
           则还会做一些特定的字符串替换处理。
        2. 对于不包含 "func" 的代码片段，它会直接计算等号右边的表达式，
           并将计算结果替换到原始字符串中，同时也会进行一些类型转换
           （如将浮点数转为整数）。
    """
    try:
        # 尝试从输入字符串中找到形如 "<<...>>" 的模式
        s = re.findall(r'<<(.*?)>>', inputs, re.DOTALL)
        # 如果没有找到匹配项，则直接返回原始输入
        if not s:
            #print("err inputs: ", origin_inputs, flush=True)
            return inputs

        index = 0
        # 遍历所有匹配到的模式
        for k in s:
            try:
                # 检查模式内是否包含 "func"
                if "func" in k:
                    # 分割并处理函数定义
                    var = k.split("=", 1)
                    try:
                        # 去除空白字符并执行函数定义
                        var[1] = var[1].strip(" ")
                        exec(var[1], globals())
                        # 调用函数获取结果
                        ans = func()
                    except:
                        # 特殊处理包含 'sympy' 的情况
                        if 'sympy' in var[1]:
                            var[1] = var[1].replace('res[x]', 'res[0][0]').replace('res[y]', 'res[0][1]')
                            exec(var[1], globals())
                            ans = func()
                        pass
                    var_list = [c.strip(" ") for c in var[0].split(",")]
                    # 如果只有一个变量名，则将结果放入列表
                    if len(var_list) == 1:
                        ans = [ans]

                    # 将结果转换为浮点数或整数形式，并替换到输入字符串中
                    for i in range(len(ans)):
                        try:
                            ans[i] = float(ans[i])
                            if abs(ans[i] - int(ans[i])) < 1e-10:
                                ans[i] = str(int(ans[i]))
                        except:
                            pass

                    # 替换原字符串中的模式和变量名
                    inputs = inputs.replace("<<"+k+">>", "")
                    for i in range(len(var_list)):
                        inputs = inputs.replace(var_list[i], str(ans[i]))
                    index += 1
                    # 更新后续模式中的变量值
                    for c in range(index, len(s)):
                        for i in range(len(var_list)):
                            s[c] = s[c].replace(var_list[i], str(ans[i]))
                else:
                    # 处理非函数的情况，直接计算并替换
                    var = k.replace(" ", "").split("=")
                    var[1] = var[1].replace("eval", "")
                    ans = round(eval(var[1]), 10)
                    ans = float(ans)
                    if abs(ans - int(ans)) < 1e-10:
                        ans = str(int(ans))
                    # 替换原字符串中的模式和变量名
                    inputs = inputs.replace("<<"+k+">>", "").replace(var[0], str(ans))
                    index += 1
                    # 更新后续模式中的变量值
                    for c in range(index, len(s)):
                        s[c] = s[c].replace(var[0], str(ans))
            except:
                return inputs
    except Exception as e:
        return inputs 

    return inputs


class TextIterStreamer:
    """ 实现文本的流式处理
        能够逐个或逐段生成和输出文本，而不是一次性输出全部内容
    """
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False, use_pot=True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        # 使用队列来缓存生成的文本片段，以便于逐块输出
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True
        # 是否使用特定的后处理技术（例如翻译或优化），默认为True
        self.use_pot = use_pot

    def put(self, value):
        # 接收并处理生成的token值
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            tokens_str = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens, errors='ignore')
            if self.use_pot:
                tokens_str = parse_pot_no_stream(tokens_str)
            self.text_queue.put(tokens_str)

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        # 实现迭代器的下一步方法，从队列中获取并返回文本，
        # 或在无更多内容时抛出StopIteration异常
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


class OutputRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`OutputLogitsProcessor`] that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.

    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
            tokens. Between 0.0 and 1.0 rewards previously generated tokens.
    """

    def __init__(self, input_length: int, 
                    presence_penalties: float = 1.0,
                    frequency_penalties: float = 0,
                    repetition_penalties: float = 0):
        if not (repetition_penalties > 0):
            raise ValueError(f"`repetition_penalties` has to be a strictly positive float, but is {repetition_penalties}")
        if not ( (frequency_penalties >= -2) and (frequency_penalties <= 2) ):
            raise ValueError(f"`frequency_penalties` has to be [-2, 2], but is {frequency_penalties}")
        if not ( (presence_penalties >= -2) and (presence_penalties <= 2) ):
            raise ValueError(f"`presence_penalties` has to be [-2, 2], but is {presence_penalties}")

        self.repetition_penalties = repetition_penalties
        self.frequency_penalties = frequency_penalties
        self.presence_penalties = presence_penalties
        self.input_length = input_length

    def _get_bin_counts_and_mask(
        self,
        tokens: torch.Tensor,
        vocab_size: int,
        num_seqs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the bin counts for the tokens.
        # vocab_size + 1 for padding.
        bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                                dtype=torch.long,
                                device=tokens.device)
        bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
        bin_counts = bin_counts[:, :vocab_size]
        mask = bin_counts > 0

        return bin_counts, mask

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        prompt_tokens_tensor = input_ids[:, :self.input_length+1]
        output_tokens_tensor = input_ids[:, self.input_length+1:]

        num_seqs, vocab_size = logits.shape
        _, prompt_mask = self._get_bin_counts_and_mask(
            prompt_tokens_tensor, vocab_size, num_seqs)
        output_bin_counts, output_mask = self._get_bin_counts_and_mask(
            output_tokens_tensor, vocab_size, num_seqs)

        repetition_penalties = torch.Tensor([self.repetition_penalties]).to(logits.device)
        frequency_penalties = torch.Tensor([self.frequency_penalties]).to(logits.device)
        presence_penalties = torch.Tensor([self.presence_penalties]).to(logits.device)

        repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
        repetition_penalties[~(prompt_mask | output_mask)] = 1.0
        logits = torch.where(logits > 0, logits / repetition_penalties,
                            logits * repetition_penalties)

        # We follow the definition in OpenAI API.
        # Refer to https://platform.openai.com/docs/api-reference/parameter-details
        logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
        logits -= presence_penalties.unsqueeze_(dim=1) * output_mask

        return logits