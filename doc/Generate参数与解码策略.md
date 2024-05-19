# Generate参数与解码策略

LLM大语言模型Generate/Inference生成或者说推理时，有很多的参数和解码策略，比如OpenAI在提供GPT系列的模型时，就提供了很多的参数，那这些参数的原理以及代码上怎么实现的呢？本文将尽力进行一一的解释。

## 1.原始生成

假如都没有使用这些参数和策略做后处理，模型是怎么生成的呢？以llama模型举例（其他生成式模型是一样）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "llama-2-7b-hf" # 模型的地址
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")

```

```bash
# 结果
inputs:{'input_ids': tensor([[   1, 1827]]), 'attention_mask': tensor([[1, 1]])}
```

输入的模型的就一个词：say（该词分词后就是1个token，在词表中的位置是1827），然后给到模型预测下一个token，如果不做任何参数控制，就是直接走模型的forward：

```python
logits = model.forward(input_ids)
print("Logits Shape:", logits.logits.shape)
print(f"logits:{logits.logits}")

```

```bash
# 结果
Logits Shape: torch.Size([1, 2, 32000])
logits:tensor([[[-12.9696,  -7.4491,  -0.4354,  ...,  -6.8250,  -8.0804,  -7.5782],
         [-11.3775, -10.1338,  -2.3563,  ...,  -6.7709,  -6.5252,  -8.9753]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
```

logits就是模型的最后输出，是一个张量（tensor），它的维度是`[batch_size, sequence_length, vocab_size]`，在这里，batch\_size是1，sequence\_length是2，只输入了一个say（一个token），那为什么是2个token呢，是由于输入模型前 llama tokenizer自动添加一个bos token ——`<s>` （开始符）, 实际输入长度就是2个token（`<s> + say`） ，llama在推理过程并没有增加（改变）输入序列的长度，最后一个token的 logits 输出预测下一个token的概率，vocab\_size是词典的大小，llama是32000，也就是说咱们接下来要在第2个sequence里找到在32000词表中哪个token的概率最大：

```python
# 在最后一个维度上（32000）进行求最大值操作，返回具有最高分数的词汇索引
next_token = torch.argmax(logits.logits, dim=-1).reshape(-1)[1]
print(f"next_token:{next_token}")

```

```bash
# 结果
next_token:tensor([22172], device='cuda:0')
```

在最后一个维度上（32000）进行求最大值操作，并返回具有最高分数的词汇索引，在词表中的位置是22172，接下来就是解码该token：

```python
next_word = tokenizer.decode(next_token)
print(f"next_word:{next_word}")

```

```bash
# 结果
next_word:hello
```

这就将next\_word预测了出来，后面的流程就是将“hello”加到“say”后面变成“say hello”，迭代上述流程直到生成eos\_token（终止词），整个预测也就完成了，这就是整个自回归的过程。上述就是不加任何参数和后处理的生成式模型的generate/inference全过程，这个过程也叫做greedy decoding贪心解码策略，下文会介绍。

## 2.常见参数（Huggingface中的常用参数[\[2\]](#ref_2 "\[2]")）

### （1）temperature

该参数用于**控制生成文本的随机性和多样性，其实是调整了模型输出的logits概率分布**，实现原理很简单，举一个简单的例子，假设我们有一个大小为`[1, 4]`的logits张量，在上述原始生成例子中其实是`[1, 32000]`，然后将logits输入到softmax函数中，分别计算没有temperature，以及temperature为0.5和temperature为2的情况下的概率分布：

```python
import torch
logits = torch.tensor([[0.5, 1.2, -1.0, 0.1]])
# 无temperature
probs = torch.softmax(logits, dim=-1)
# temperature low 0.5
probs_low = torch.softmax(logits / 0.5, dim=-1)
# temperature high 2
probs_high = torch.softmax(logits / 2, dim=-1)

print(f"probs:{probs}")
print(f"probs_low:{probs_low}")
print(f"probs_high:{probs_high}")

```

```bash
# 结果
probs: tensor([[0.2559, 0.5154, 0.0571, 0.1716]])
probs_low: tensor([[0.1800, 0.7301, 0.0090, 0.0809]])
probs_high: tensor([[0.2695, 0.3825, 0.1273, 0.2207]])
```

从上述结果中可以看到，**当temperature较高时，会更平均地分配概率给各个token，这导致生成的文本更具随机性和多样性；temperature较低接近0时，会倾向于选择概率最高的token，从而使生成的文本更加确定和集中**。注：temperature=1时表示不使用此方式。

### （2）top\_p

top-p也是一个用于控制生成文本多样性的参数，也被称为"nucleus sampling"。这个参数的全名是"top probability"，**通常用一个介于 0 到 1 之间的值来表示生成下一个token时，在概率分布中选择的最高概率的累积阈值**，看一下是怎么实现的，还以上述的例子为例：

```python
import torch
# 样例：probs: tensor([[0.2559, 0.5154, 0.0571, 0.1716]])
probs = torch.tensor([[0.2559, 0.5154, 0.0571, 0.1716]])
# 第一步进行排序
probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
# 结果
probs_sort: tensor([[0.5154, 0.2559, 0.1716, 0.0571]])
probs_idx: tensor([[1, 0, 3, 2]])

# 第二步概率的累积和
probs_sum = torch.cumsum(probs_sort, dim=-1)
# 结果
probs_sum: tensor([[0.5154, 0.7713, 0.9429, 1.0000]])

# 第三步找到第一个大于阈值p的位置，假设p=0.9，并将后面的概率值置为0：
mask = probs_sum - probs_sort > p
probs_sort[mask] = 0.0
# 结果
probs_sort: tensor([[0.5154, 0.2559, 0.1716, 0.0000]])

# 第四步复原原序列
new_probs = probs_sort.scatter(1, probs_idx, probs_sort)
# 结果
new_probs: tensor([[0.2559, 0.5154, 0.0000, 0.1716]])

# 注：在真实实现中一般会把舍弃的概率置为-inf，即
zero_indices = (new_probs == 0)
new_probs[zero_indices] = float('-inf')
# 结果
new_probs: tensor([[0.2559, 0.5154, -inf, 0.1716]])

```

```python
# 完整代码
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    new_probs = probs_sort.scatter(1, probs_idx, probs_sort)
    zero_indices = (new_probs == 0)
    new_probs[zero_indices] = float('-inf')
    return new_probs
```

从上述的实现中可以看到，当top\_p较高时比如 0.9，这意味着前 90% 的概率的token会被考虑在抽样中，这样会允许更多的token参与抽样，增加生成文本的多样性；当top\_p较低时比如比如 0.1，这意味着只有前 10% 最高概率的token会被考虑在抽样中，这样会限制生成文本的可能性，使生成的文本更加确定和集中。这里可能会有一点疑问，当top-p设置的很小，累加的概率没超过怎么办？一般代码中都会强制至少选出一个token的。注：top\_p=1时表示不使用此方式。

### （3）top\_k

这个参数比较简单，简单来说就是用于在**生成下一个token时，限制模型只能考虑前k个概率最高的token**，这个策略可以降低模型生成无意义或重复的输出的概率，同时提高模型的生成速度和效率。实现如下：

```python
import torch
filter_value = -float("Inf")
top_k = 2
probs = torch.tensor([[0.2559, 0.5154, 0.0571, 0.1716]])
indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
new_probs = probs.masked_fill(indices_to_remove, filter_value)
print("new_probs:", new_probs)

```

```bash
# 结果
new_probs: tensor([[0.2559, 0.5154,   -inf,   -inf]])
```

### （4）repetition\_penalty

这个重复惩罚参数也比较容易理解，**通过修改生成文本时的概率分布来实现的**， repetition\_penalty的目标是在这个概率分布中对先前生成过的token，又重复的生成了该token进行惩罚（降低概率），以减少生成文本中的重复性，简单实现如下：

```python
import numpy as np
def apply_repetition_penalty(probs, repetition_penalty, prev_tokens):
    adjusted_probs = np.copy(probs)
    for token in set(prev_tokens):
        adjusted_probs[token] *= (1/repetition_penalty)
    adjusted_probs /= np.sum(adjusted_probs)  
    return adjusted_probs
# 示例概率分布，索引对应不同词语
original_probs = np.array([0.3, 0.1, 0.3, 0.1, 0.2])
# 示例先前生成的词语
previous_tokens = [2, 4, 2]
# 重复惩罚系数
repetition_penalty = 1.25
# 应用重复惩罚，得到调整后的概率分布
adjusted_probs = apply_repetition_penalty(original_probs, repetition_penalty, previous_tokens)

print("原始概率分布：", original_probs)
print("调整后的概率分布：", adjusted_probs)

```

```bash
# 结果
原始概率分布： [0.3 0.1 0.3 0.1 0.2]
调整后的概率分布： [0.33333333 0.11111111 0.26666667 0.11111111 0.17777778]
```

上述结果很明显看的出来，出现过的的token概率变低了，未出现过的token的概率变高了。注：repetition\_penalty=1时表示不进行惩罚。

### （5）no\_repeat\_ngram\_size

这个参数，**当设为大于0的整数时，生成的文本中不会出现指定大小的重复n-gram**（n个连续的token），可以使生成的文本更加多样化，避免出现重复的短语或句子结构。实现原理和上述repetition\_penalty的是大致差不多的，只不过这里是n个连续的token。注默认不设置

### （6）do\_sample

这个参数是**对模型计算出来的概率要不要进行多项式采样**，多项式采样（Multinomial Sampling）是一种用于**从一个具有多个可能结果的离散概率分布中进行随机抽样的方法**，多项式采样的步骤如下：

1. 首先，根据概率分布对应的概率，为每个可能结果分配一个抽样概率。这些抽样概率之和必须为1。
2. 然后，在进行一次抽样时，会根据这些抽样概率来选择一个结果。具体地，会生成一个随机数，然后根据抽样概率选择结果。抽样概率越高的结果，被选中的概率也就越大。
3. 最终，被选中的结果就是这次抽样的输出。

在多项式采样中，**概率高的结果更有可能被选中，但不同于确定性的选择，每个结果仍然有一定的概率被选中**。这使得模型在生成文本时具有一定的随机性，但又受到概率的控制，以便生成更加多样且符合概率分布的文本。实现如下：

```python
import torch
probs = torch.tensor([[0.2559, 0.5154, 0.0571, 0.1716]])
next_token = torch.multinomial(probs, num_samples=1)
print("next_token:", next_token)
# 结果
next_token: tensor([[1]])
```

这个do\_sample参数通过多样式采样会有一定的随机性，这种随机性导致了生成的文本更加多样化，因为模型有机会选择概率较低但仍然可能的词，这种方法可以产生丰富、有趣、创新的文本，但可能会牺牲一些文本的准确性。注do\_sample=False，不进行采样。在Huggingface中，do\_sample这个参数有更高的含义即做不做多样化采样，**当do\_sample=False时，temperature，top\_k，top\_p这些参数是不能够被设置的，只有do\_sample=True时才能够被设置**。比如：

```python
UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.5` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.1` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
```

### （7）num\_beams

num\_beams参数是用于束搜索（beam search）算法的，其用途是**控制生成的多个候选句子的数量，该参数控制的是每个生成步要保留的生成结果的数量**，用于在生成过程中增加多样性或生成多个可能的结果。主要步骤如下：

1. 在每个生成步，对于前一个生成中的所有生成结果，分别基于概率保留前 k 个最可能的结果（k 即 num\_beams 参数的值）。
2. 将所有扩展后的生成结果，按照其对应的概率分数重新计算分数并进行排序，并保留前 k 个最可能的结果。
3. 如果已经生成了结束符，则将其对应的结果保留下来。
4. 重复上述过程直到生成所有的结果或达到最大长度。

以下是简单代码实现，结合代码会更容易理解：

首先定义一个BeamSearchNode类

```python
class BeamSearchNode:
    def __init__(self, sequence, score):
        self.sequence = sequence  # 生成的序列
        self.score = score  # 分数（概率）
```

然后给出接下来生成token的概率，简单起见给一个固定的概率

```python
# 示例：下一个token的概率函数，简单使用固定概率
def simple_next_word_probs(sequence):
    if sequence[-1] == "<end>":
        return {}
    return {"apple": 0.3, "like": 0.35, "peach": 0.2, "banana": 0.15}
```

下面是beam\_search算法的非常简单的实现，工程上实现的话参考Huggingface的官方实现，

```python
def beam_search(initial_sequence, next_word_probs_func, num_beams, max_sequence_length):
    # 初始化初始节点，且分数为1
    initial_node = BeamSearchNode(sequence=initial_sequence, score=1.0)
    candidates = [initial_node]

    final_candidates = []  # 最终的候选序列
    # 只要候选节点列表不为空，且 final_candidates 中的候选节点数量还没有达到指定的束宽度，就继续进行搜索
    while candidates and len(final_candidates) < num_beams:
        # 候选节点排序
        candidates.sort(key=lambda x: -x.score)
        current_node = candidates.pop(0)
        # 当节点序列末尾生成结束符号（如"<end>"），或者当生成的序列长度达到最大限制时终止节点的扩展
        if current_node.sequence[-1] == "<end>" or len(current_node.sequence) >= max_sequence_length:
            final_candidates.append(current_node)
        else:
            # 获取下一个token的概率，我们的例子返回的是固定的概率
            next_words_probs = next_word_probs_func(current_node.sequence) 
            # 生成新的候选序列，并计算分数           
            for next_word, next_word_prob in next_words_probs.items():
                new_sequence = current_node.sequence + [next_word]
                new_score = current_node.score * next_word_prob
                new_node = BeamSearchNode(sequence=new_sequence, score=new_score)
                candidates.append(new_node)

    return [candidate.sequence for candidate in final_candidates]
```

开始使用：

```python
initial_sequence = ["<start>", "I"]
num_beams = 3
max_sequence_length = 3
result = beam_search(initial_sequence, simple_next_word_probs, num_beams, max_sequence_length)

for idx, sequence in enumerate(result):
    print(f"Sentence {idx + 1}: {' '.join(sequence)}")

# 结果，符合我们的概率分布
Sentence 1: <start> I like
Sentence 2: <start> I apple
Sentence 3: <start> I peach

# 当长度为4时呢？
Sentence 1: <start> I like like
Sentence 2: <start> I like apple
Sentence 3: <start> I apple like

# 为什么结果变了呢？我们来看一下最终每个序列的分数
current_node: ['<start>', 'I', 'like', 'like']
current_score: 0.12249999999999998
current_node: ['<start>', 'I', 'like', 'apple']
current_score: 0.105
current_node: ['<start>', 'I', 'apple', 'like']
current_score: 0.105

# 再看一下其他序列的分数，apple的
new_node: ['<start>', 'I', 'apple', 'apple']
new_score: 0.09
new_node: ['<start>', 'I', 'apple', 'like']
new_score: 0.105
new_node: ['<start>', 'I', 'apple', 'peach']
new_score: 0.06
new_node: ['<start>', 'I', 'apple', 'banana']
new_score: 0.045

# 再看一下其他序列的分数，peach的
new_node: ['<start>', 'I', 'peach', 'apple']
new_score: 0.06
new_node: ['<start>', 'I', 'peach', 'like']
new_score: 0.06999999999999999
new_node: ['<start>', 'I', 'peach', 'peach']
new_score: 0.04000000000000001
new_node: ['<start>', 'I', 'peach', 'banana']
new_score: 0.03
```

上述就是beam search的简单代码实现，有优点也有相应的缺点

优点：

1. **生成多样性**\*\*：\*\*  通过增加num\_beams束宽，束搜索可以保留更多的候选序列，从而生成更多样化的结果。
2. **找到较优解**\*\*：\*\*  增加num\_beams束宽有助于保留更多可能的候选序列，从而更有可能找到更优的解码结果，这在生成任务中有助于避免陷入局部最优解
3. **控制输出数量**\*\*：\*\*  通过调整num\_beams束宽，可以精确控制生成的候选序列数量，从而平衡生成结果的多样性和数量。

缺点：

1. **计算复杂度**\*\*：\*\*  随着num\_beams束宽的增加，计算复杂度呈指数级增长，较大的束宽会导致解码过程变得更加耗时，尤其是在资源有限的设备上。
2. **忽略概率较低的序列**\*\*：\*\*  增加num\_beams束宽可能会导致一些低概率的候选序列被忽略，因为搜索过程倾向于集中在概率较高的路径上，从而可能错过一些潜在的优质解。
3. **缺乏多样性**\*\*：\*\*  尽管增加num\_beams束宽可以增加生成结果的多样性，但束搜索仍然可能导致**生成的结果过于相似**，因为它倾向于选择概率较高的路径。

### （8）num\_beam\_groups

Huggingface中的生成参数中也是有该参数的，这个参数背后其实是一种beam search算法的改进，叫做Diverse Beam Search (DBS)，上述已经讨论了beam search生成的结果还是会过于相似的，这个DBS做了一些改进，**核心就是分组机制**，举个例子来说如果我的num\_beams=2，num\_beam\_groups=2，那就是说分成2个组，每个组里的beam可以相似，但组和组之间要有足够的多样性，引入了多样性分数，具体实现细节可以看一下论文，不过以下一张图就容易理解了：

![](image/image_w13y9FgYsi.png)

### （9）diversity\_penalty

这个多样性惩罚参数只有在启用了“num\_beam\_groups”（组束搜索）时才有效，在这些组之间应用多样性惩罚，以确保每个组生成的内容尽可能不同。

### （10）length\_penalty

这个长度惩罚参数也是用于束搜索过程中，在束搜索的生成中，候选序列的得分通过对数似然估计计算得到，即得分是负对数似然。l**ength\_penalty的作用是将生成序列的长度应用于得分的分母，从而影响候选序列的得分**，当length\_penalty > 1.0时，较长的序列得到更大的惩罚，鼓励生成较短的序列；当length\_penalty< 1.0时，较短的序列得到更大的惩罚，鼓励生成较长的序列，默认为1，不受惩罚。

### （11）use\_cache

该参数如何设置为True时，则模型会利用之前计算得到的注意力权重（key/values attentions）的缓存，这些注意力权重是在模型生成文本的过程中，根据输入上下文和已生成部分文本，计算出来的，当下一个token需要被生成时，模型可以通过缓存的注意力权重来重用之前计算的信息，而不需要重新计算一次，有效地跳过重复计算的步骤，从而减少计算负担，提高生成速度和效率。

## 3.其他参数

接下来对比较简单和少见的参数做一下简单阐述

### （1）num\_return\_sequences

该参数是模型返回不同的文本序列的数量，要和beam search中的num\_beams一致，在贪心解码策略中（下述会讲到），num\_return\_sequences只能为1，默认也为1。

### （2）max\_length

生成的token的最大长度。它是输入prompt的长度加上max\_new\_tokens的值。如果同时设置了max\_new\_tokens，则会覆盖此参数，默认为20。

### （3）max\_new\_tokens

生成的最大token的数量，不考虑输入prompt中的token数，默认无设置

### （4）min\_length

生成的token的最小长度。它是输入prompt的长度加上min\_new\_ tokens的值。如果同时设置了min\_new\_tokens，则会覆盖此参数，默认为0。

### （5）min\_new\_tokens

生成的最小token的数量，不考虑输入prompt中的token数，默认无设置

### （6）early\_stopping

控制基于束搜索（beam search）等方法的停止条件，接受以下值：

- True：生成会在出现num\_beams个完整候选项时停止。
- False：应用启发式方法，当很不可能找到更好的候选项时停止生成。
- never：只有当不能找到更好的候选项时，束搜索过程才会停止（经典的束搜索算法）。

默认为False

### （7）bad\_words\_ids

包含词汇id的列表，这个参数用于指定不允许在生成文本中出现的词汇,如果生成的文本包含任何在这个列表中的词汇，它们将被被替换或排除在最终生成的文本之外。

### （8）force\_words\_ids

包含词汇id的列表，用于指定必须包含在生成文本中的词汇，如果给定一个列表，生成的文本将包含这些词汇。

### （9）constraints

自定义约束条件，可以指定约束条件，这些约束条件可以是必须出现的关键词、短语、特定术语或其他文本元素，其实和force\_words\_ids是差不多的意思，在代码实现也是一样的。

## 4.常见解码策略（Huggingface中实现的解码策略）

Huggingface如何判断采用那种解码策略呢？如果直接使用`model.generate()`，就是上述参数的组合，会用来一一判断使用那种解码策略，如果出现冲突会抛出异常。目前Huggingface总共有8种解码策略，从Huggingface的判断顺序来说起：

### （1）constrained beam-search decoding

受限束搜索解码，使用`model.generate()`当 constraints 不为 None 或 force\_words\_ids不为 None 时进入该模式，而且要求num\_beams要大于1（本质还是束搜索），do\_sample为False，num\_beam\_groups为1，否则就会抛出：

```python
"`num_beams` needs to be greater than 1 for constrained generation."
"`do_sample` needs to be false for constrained generation."
"`num_beam_groups` not supported yet for constrained generation."
```

在这个解码策略中，核心还是上述实现的beam search，只不过在search中加入了提供的词表，强制其生成提供的词表，来看一下怎么使用，先来看一下传统的beam search使用：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
model_name = "llama-2-7b-hf" # 模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

# generate实现
generation_output = model.generate(
    input_ids=input_ids,
    num_beams = 3,
    num_return_sequences=3,
    return_dict_in_generate=True,
    max_new_tokens=3,
)

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

```

```bash
# 结果
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to your new favorite
Generated sequence 2: say hello to your new best
Generated sequence 3: say hello to our newest
```

加上了约束之后，即给定词表`["my"]`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
model_name = "llama-2-7b-hf" # 你模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

force_words = ["my"]
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

generation_output = model.generate(
    input_ids=input_ids,
    force_words_ids = force_words_ids,
    num_beams = 3,
    num_return_sequences=3,
    return_dict_in_generate=True,
    max_new_tokens=3,
)

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

```

```bash
# 结果    
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to my little friend
Generated sequence 2: say hello to your new favorite
Generated sequence 3: say hello to your new my
```

结果很明显，生成中出现了限制词表“my”。

### （2）contrastive search

对比搜索策略，在`model.generate()`中，当 penalty\_alpha 大于 0 且top\_k>1大于 1 时使用，这是一种引入对比度惩罚的搜索方法，之前没有介绍过penalty\_alpha这个惩罚因子参数，因为只有在contrastive search是才会用到。这种解码策略是在2022年A Contrastive Framework for Neural Text Generation论文中提出来的方法，具体细节可以看论文，Huggingface已经实现，来看一下简单的原理：生成的token应该是从模型预测的最佳候选（top k）中而来；在生成token时，当前token应该能与前面生成的内容保持对比性（或差异性），其实现就是若当前生成的token 与之前的序列token相似度很大，就减少其整体概率值，进而减少它被解码出来的可能性，避免重复解码的问题。

核心公式如下：

$$
x_{t}=\underset{v \in V^{(k)}}{\arg \max }\{(1-\alpha) \times \underbrace{p_{\theta}\left(v \mid \boldsymbol{x}_{<t}\right)}_{\text {model confidence }}-\alpha \times \underbrace{\left(\max \left\{s\left(h_{v}, h_{x_{j}}\right): 1 \leq j \leq t-1\right\}\right)}_{\text {degeneration penalty }}\},
$$

其中$V^{(k)}$是token$x_t$候选集合，是根据模型$p_\theta(v|x_{<t})$预测的top-k tokens，k一般取值3\~10；上述公式表示要生成的$x_t$来自$V^{(k)}$集合中概率最大的那个token；每个候选tokenv的概率计算分两部分：

1. $p_\theta(v|x_{<t})$为模型预测的概率，可以保证一定流畅性；
2. $max\{s(h_v,h\_{x_j}):1<=j<=t-1\}$为token v跟之前生成的序列token相似度中取最大的那个值。

核心实现代码如下：

```python
def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, alpha):
    '''
       该函数是实现Contrastive Search中next token预测中候选token的排序分数，分数最大对应token为输出结果
        context_hidden: beam_width x context_len x embed_dim ,用于计算相似度，是公式中x_j集合表征向量
        next_hidden: beam_width x 1 x embed_dim，用于计算相似度，是公式中候选token v 的表征向量
        next_top_k_ids: beam_width x 1，记录候选token的编码
        next_top_k_probs，候选token的模型预测概率
        alpha，惩罚参数
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True) 
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1) #计算相似度矩阵
    assert cosine_matrix.size() == torch.Size([beam_width, context_len])
    scores, _ = torch.max(cosine_matrix, dim = -1) #输出公式第二项值
    assert scores.size() == torch.Size([beam_width])
    next_top_k_probs = next_top_k_probs.view(-1)  #输出公式第一项值
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores  #对应公式整体计算
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    return next_id
```

如何使用：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
model_name = "llama-2-7b-hf" # 你模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

generation_output = model.generate(
    input_ids=input_ids,
    penalty_alpha = 0.5,
    top_k = 30,
    return_dict_in_generate=True,
    max_new_tokens=3,
)

# 直接使用其函数
# generation_output = model.contrastive_search(
#     input_ids=input_ids,
#     penalty_alpha = 0.5,
#     top_k = 30,
#     return_dict_in_generate=True,
#     max_new_tokens=3,
# )

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

```

```bash
# 结果
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to 20
```

### （3）greedy decoding

最经典最原始的贪心解码策略，在`model.generate()`中，当 num\_beams 等于 1 且 do\_sample 等于 False 时进入此模式，也可以直接使用`model.greedy_search()`，这个解码策略很简单，就是在每一步中选择预测概率最高的token作为下一个token，从而生成文本，和之前的forword是一样的，这种方法通常会导致生成的文本比较单一和局部最优。注意此策略不能使用temperature，top\_k，top\_p等改变logits的参数。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
model_name = "llama-2-7b-hf" # 你模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

generation_output = model.generate(
    input_ids=input_ids,
    num_beams = 1,
    do_sample = False,
    return_dict_in_generate=True,
    max_new_tokens=3,
)
# 直接指定使用其函数
# generation_output = model.greedy_search(
#     input_ids=input_ids,
#     num_beams = 1,
#     do_sample = False,
#     return_dict_in_generate=True,
#     max_length = 7
# )

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

```

```bash
# 结果
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to the newest
```

### （4）multinomial sampling

多项式采样解码策略，在`model.generate()`中，当 num\_beams 等于 1 且 do\_sample 等于 True 时进入此模式，也可以使用`model.sample()`，该策略通过各种改变logits的参数（multinomial sampling，temperature，top\_k，top\_p等）从而实现生成文本的多样性。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    )

import torch
model_name = "llama-2-7b-hf" # 你模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

generation_output = model.generate(
    input_ids=input_ids,
    num_beams = 1,
    do_sample = True,
    temperature = 1.2,
    top_k = 100,
    top_p = 0.6,
    return_dict_in_generate=True,
    max_length=7,
)

# sample实现
# logits_warper = LogitsProcessorList(
#     [
#     TopKLogitsWarper(100),
#     TemperatureLogitsWarper(1.2),
#     TopPLogitsWarper(0.6)
#     ]
# )
# generation_output = model.sample(
#     input_ids=input_ids,
#     logits_warper=logits_warper,
#     return_dict_in_generate=True,
#     max_length=7,
# )

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

```

```bash
# 注意这种方式每次结果都可能不一样
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to our new intern
```

### （5）beam-search decoding

beam search的解码策略，上述已经讲解过实现过程，在`model.generate()`中是当 num\_beams 大于 1 且 do\_sample 等于 False 时使用，也可以调用`model.beam_search()` 来实现，在此就不过多的赘述。

### （6）beam-search multinomial sampling

beam-search中在实现采样的方式，其实就是在`model.generate()`中，当 num\_beams 大于 1 且 do\_sample 等于 True 时使用，其实就是在beam search中加入了多样化的采样方式，在此就不过多的赘述。

### （7）diverse beam-search decoding

分组的beam-search解码方式，上述在解释`num_beam_groups`，已经进行过介绍，在`model.generate()`中，当 num\_beams 大于 1 ， num\_beam\_groups 大于 1 ，diversity\_penalty大于0，do\_sample 等于 False 时进入此模式，在此就不过多的赘述。

### （8）assisted decoding

这一种解码方式比较有意思，叫做辅助解码，意思是使用另一个模型（称为辅助模型）的输出来辅助生成文本，一般是借助较小的模型来加速生成候选 token，辅助模型必须具有与目标模型完全相同的分词器（tokenizer），来简单实现一下，通过llama7B辅助生成llama13B，一般来说辅助模型要很小，这里只是简单实验：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
model_name = "llama-2-13b-hf" # 你自己模型的位置
assistant_model_name = "llama-2-7b-hf" # 你自己模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "say hello to"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
input_ids = inputs["input_ids"].to("cuda")

generation_output = model.generate(
    assistant_model=assistant_model,
    input_ids=input_ids,
    num_beams = 1,
    do_sample = False,
    return_dict_in_generate=True,
    max_length=7,
)

print("query:", text)
for i, output_sequence in enumerate(generation_output.sequences):
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Generated sequence {i+1}: {output_text}")

# 结果
inputs:{'input_ids': tensor([[    1,  1827, 22172,   304]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
query: say hello to
Generated sequence 1: say hello to the newest
```

上述就是Huggingface中常用的参数，以及目前实现的解码策略的简单原理介绍和一些代码实现，目前关于解码策略方向的研究工作还是比较热的，因为并不是每个实验室都用8张以上A100。
