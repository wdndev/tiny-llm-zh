# Trainer参数

本文主要记录一下**Transformers的Trainer 以及其训练参数**，主要参考的就是官网的文档，版本为4.34.0，这些参数的顺序也是按照官网的顺序来的，简单的参数就直接翻译了一下。

## **1.Transformers的Trainer类接受的参数：**&#x20;

1. **`model`**\*\* (****`PreTrainedModel`**** 或 ****`torch.nn.Module`****, 可选) \*\*：要进行训练、评估或预测的实例化后模型，如果不提供，必须传递一个`model_init`来初始化一个模型。
2. **`args`**\*\* (****`TrainingArguments`****, 可选) \*\*：训练的参数，如果不提供，就会使用默认的`TrainingArguments` 里面的参数，其中 `output_dir` 设置为当前目录中的名为 "tmp\_trainer" 的目录。
3. **`data_collator`**\*\* (****`DataCollator`****, 可选) \*\*：用于从`train_dataset` 或 `eval_dataset` 中构成batch的函数，如果未提供tokenizer，将默认使用 `default_data_collator()`；如果提供，将使用 `DataCollatorWithPadding` 。
4. **`train_dataset`**\*\* (****`torch.utils.data.Dataset`**** 或 ****`torch.utils.data.IterableDataset`****, 可选) \*\*：用于训练的数据集，如果是torch.utils.data.Dataset，则会自动删除模型的`forward()` 方法不接受的列。
5. **`eval_dataset`**\*\* (Union\[torch.utils.data.Dataset, Dict\[str, torch.utils.data.Dataset]), 可选)\*\*：同上，用于评估的数据集，如果是字典，将对每个数据集进行评估，并在指标名称前附加字典的键值。
6. **`tokenizer`**\*\* (PreTrainedTokenizerBase, 可选)\*\*：用于预处理数据的分词器，如果提供，将在批量输入时自动对输入进行填充到最大长度，并会保存在模型目录下中，为了重新运行中断的训练或重复微调模型时更容易进行操作。
7. **`model_init`**\*\* (Callable\[\[], PreTrainedModel], 可选)\*\*：用于实例化要使用的模型的函数，如果提供，每次调用 `train()` 时都会从此函数给出的模型的新实例开始。
8. **`compute_metrics`**\*\* (Callable\[\[EvalPrediction], Dict], 可选)\*\*：用于在评估时计算指标的函数，必须接受 `EvalPrediction` 作为入参，并返回一个字典，其中包含了不同性能指标的名称和相应的数值，一般是准确度、精确度、召回率、F1 分数等。
9. **`callbacks`**\*\* (TrainerCallback 列表, 可选)\*\*：自定义回调函数，如果要删除使用的默认回调函数，要使用 `Trainer.remove_callback()` 方法。
10. **`optimizers`**\*\* (Tuple\[torch.optim.Optimizer, torch.optim.lr\_scheduler.LambdaLR], 可选) \*\*：用于指定一个包含优化器和学习率调度器的元组（Tuple），这个元组的两个元素分别是优化器

    （`torch.optim.Optimizer`）和学习率调度器（`torch.optim.lr_scheduler.LambdaLR`），默认会创建一个基于AdamW优化器的实例，并使用 `get_linear_schedule_with_warmup()` 函数创建一个学习率调度器。
11. **`preprocess_logits_for_metrics`**\*\* (Callable\[\[torch.Tensor, torch.Tensor], torch.Tensor], 可选)\*\*：用于指定一个函数，这个函数在每次评估步骤（evaluation step）前，其实就是在进入compute\_metrics函数前对模型的输出 logits 进行预处理。接受两个张量（tensors）作为参数，一个是模型的输出 logits，另一个是真实标签（labels）。然后返回一个经过预处理后的 logits 张量，给到compute\_metrics函数作为参数。

## **2.TrainingArguments的参数**

1. **`output_dir`**\*\* (str)\*\*：用于指定模型checkpoint和最终结果的输出目录。 &#x20;
2. **`overwrite_output_dir`**\*\* (bool, 可选，默认为 False)**：如果设置为True，将**覆盖输出目录\*\*中已存在的内容，在想要继续训练模型并且输出目录指向一个checkpoint目录时还是比较有用的。 &#x20;
3. **`do_train`**\*\* (bool, 可选，默认为 False)\*\*：是否执行训练，其实Trainer是不直接使用此参数，主要是用于在写脚本时，作为if的条件来判断是否执行接下来的代码。 &#x20;
4. **`do_eval`**\*\* (bool, 可选)\*\*：是否在验证集上进行评估，如果评估策略（evaluation\_strategy）不是"no"，将自动设置为True。与do\_train类似，也不是直接由Trainer使用的，主要是用于我们写训练脚本。 &#x20;
5. **`do_predict`**\*\* (bool, 可选，默认为 False)\*\*：是否在测试集上进行预测。 &#x20;
6. **`evaluation_strategy `(str, 可选，默认为 "no")**：用于指定训练期间采用的评估策略，可选值包括： &#x20;
   - "no"：在训练期间不进行任何评估。 &#x20;
   - "steps"：每eval\_steps步骤进行评估。 &#x20;
   - "epoch"：在每个训练周期结束时进行评估。
7. **`prediction_loss_only `(bool, 可选, 默认为 False)**：如果设置为True，当进行评估和预测时，只返回损失值，而不返回其他评估指标。 &#x20;
8. **`per_device_train_batch_size`**\*\* (int, 可选, 默认为 8)\*\*：用于指定训练的每个GPU/XPU/TPU/MPS/NPU/CPU的batch，每个训练步骤中每个硬件上的样本数量。 &#x20;
9. **`per_device_eval_batch_size`**\*\* (int, 可选, 默认为 8)\*\*：用于指定评估的每个GPU/XPU/TPU/MPS/NPU/CPU的batch，每个评估步骤中每个硬件上的样本数量。 &#x20;
10. **`gradient_accumulation_steps`**\*\* (int, 可选, 默认为 1)\*\*：用于指定在每次更新模型参数之前，梯度积累的更新步数。使得梯度积累可以在多个batch上累积梯度，然后更新模型参数，就可以在显存不够的情况下执行大batch的反向传播。 &#x20;

    假设有4张卡，每张卡的batch size为8，那么一个steps的batch size就是32，如果我们这个参数设置为4，那么相当于一个batch训练样本数量就是128。**好处：显存不够增大此参数**。 &#x20;
11. **`eval_accumulation_steps`**\*\* (int, 可选)\*\*：指定在执行评估时，模型会累积多少个预测步骤的输出张量，然后才将它们从GPU/NPU/TPU移动到CPU上，默认是整个评估的输出结果将在GPU/NPU/TPU上累积，然后一次性传输到CPU，速度更快，但占显存。 &#x20;
12. **`eval_delay`**\*\* (float, 可选)\*\*：指定等待执行第一次评估的轮数或步数。如果evaluation\_strategy为"steps"，设置此参数为10，则10个steps后才进行首次评估。 &#x20;
13. **`learning_rate`**\*\* (float, 可选, 默认为 5e-5)\*\*：指定AdamW优化器的初始学习率。 &#x20;
14. **`weight_decay`**\*\* (float, 可选, 默认为 0)\*\*：指定权重衰减的值，会应用在 AdamW 优化器的所有层上，除了偏置（bias）和 Layer Normalization 层（LayerNorm）的权重上。 &#x20;

    简单解释一下，权重衰减是一种正则化手段，通过向损失函数添加一个额外的项来惩罚较大的权重值，有助于防止模型过拟合训练数据。 &#x20;
15. **`adam_beta1`**\*\* (float, 可选, 默认为 0.9)\*\*：指定AdamW优化器的beta1超参数，详细的解释可以看其论文。 &#x20;
16. **`adam_beta2`**\*\* (float, 可选, 默认为 0.999)\*\*：指定AdamW优化器的beta2超参数，详细的解释可以看其论文。 &#x20;
17. **`adam_epsilon`**\*\* (float, 可选, 默认为 1e-8)\*\*：指定AdamW优化器的epsilon超参数，详细的解释可以看其论文。 &#x20;
18. **`max_grad_norm`**\*\* (float, 可选, 默认为 1.0)\*\*：指定梯度剪裁的最大梯度范数，可以防止梯度爆炸，一般都是1，如果某一步梯度的L2范数超过了 此参数，那么梯度将被重新缩放，确保它的大小不超过此参数。 &#x20;
19. **`num_train_epochs`**\*\* (float, 可选, 默认为 3.0)\*\*：训练的总epochs数。 &#x20;
20. **`max_steps`**\*\* (int, 可选, 默认为 -1)\*\*：如果设置为正数，就是执行的总训练步数，**会覆盖num\_train\_epochs**。注意如果使用此参数，就算没有达到这个参数值的步数，训练也会在数据跑完后停止。 &#x20;
21. **`lr_scheduler_type`**\*\* (str, 可选, 默认为"linear")\*\*：用于指定学习率scheduler的类型，根据训练的进程来自动调整学习率。详细见： &#x20;
    - **"linear"**：线性学习率scheduler，学习率以线性方式改变 &#x20;
    - **"cosine"**：余弦学习率scheduler，学习率以余弦形状的方式改变。 &#x20;
    - **"constant"**：常数学习率，学习率在整个训练过程中保持不变。 &#x20;
    - **"polynomial"**：多项式学习率scheduler，学习率按多项式函数的方式变化。 &#x20;
    - **"piecewise"**：分段常数学习率scheduler，每个阶段使用不同的学习率。 &#x20;
    - **"exponential"**：指数学习率scheduler，学习率以指数方式改变。
22. **`warmup_ratio`**\*\* (float, 可选, 默认为0.0)\*\*：用于指定线性热身占总训练步骤的比例，线性热身是一种训练策略，学习率在开始阶段从0逐渐增加到其最大值（通常是设定的学习率），然后在随后的训练中保持不变或者按照其他调度策略进行调整。如果设置为0.0，表示没有热身。 &#x20;
23. **`warmup_steps`**\*\* (int,可选, 默认为0)\*\*：这个是直接指定线性热身的步骤数，这个参数会覆盖warmup\_ratio，如果设置了warmup\_steps，将会忽略warmup\_ratio。 &#x20;
24. **`log_level`**\*\* (str, 可选, 默认为passive)\*\*：用于指定主进程上要使用的日志级别， &#x20;
    - debug：最详细的日志级别。 &#x20;
    - info：用于一般的信息性消息。 &#x20;
    - warning：用于警告信息。 &#x20;
    - error：用于错误信息。 &#x20;
    - critical：用于严重错误信息。 &#x20;
    - passive：不设置任何内容，将会使用Transformers库当前的日志级别（默认为"warning"）。 &#x20;
      建议训练时使用info级别。
25. **`log_level_replica`**\*\* (str, 可选, 默认为warning)\*\*：副本上要使用的日志级别，与log\_level相同。 &#x20;
26. **`log_on_each_node`**\*\* (bool, optional, defaults to True)\*\*：在多节点分布式训练中，是否在每个节点上使用log\_level进行日志记录。 &#x20;
27. **`logging_dir`**\*\* (str, 可选)\*\*：TensorBoard日志目录。默认为output\_dir/runs/CURRENT\_DATETIME\_HOSTNAME。 &#x20;
28. **`logging_strategy`**\*\* (str, 可选, 默认为"steps")\*\*：训练过程中采用的日志记录策略。可选包括： &#x20;
    - "no"：在训练过程中不记录任何日志。 &#x20;
    - "epoch"：在每个epoch结束时记录日志。 &#x20;
    - "steps"：根据logging\_steps参数记录日志。
29. **`logging_steps`**\*\* (int or float,可选, 默认为500)\*\*：如果logging\_strategy="steps"，则此参数为每多少步记录一次步骤。 &#x20;
30. **`logging_nan_inf_filter`**\*\* (bool, 可选, 默认为 True)\*\*：是否过滤日志记录中为nan和inf的loss，如果设置为True，将过滤每个步骤的loss，如果出现nan或inf，将取当前日志窗口的平均损失值。 &#x20;
31. **`save_strategy`**\*\* (str , 可选, 默认为 "steps")\*\*：训练过程中保存checkpoint的策略，包括： &#x20;
    - "no"：在训练过程中不保存checkpoint。 &#x20;
    - "epoch"：在每个epoch束时保存checkpoint。 &#x20;
    - "steps"：根据save\_steps参数保存checkpoint。
32. **`save_steps`**\*\* (int or float, 可选, 默认为500)\*\*：如果save\_strategy="steps"，就是指两次checkpoint保存之间的更新步骤数。如果是在\[0, 1)的浮点数，则就会当做与总训练步骤数的比例。
33. **`save_total_limit`**\*\* (int, 可选)\*\*：如果给定了参数，将限制checkpoint的总数，因为checkpoint也是很占硬盘的，将会删除输出目录中旧的checkpoint。当启用load\_best\_model\_at\_end时，会根据metric\_for\_best\_model保留最好的checkpoint，以及最近的checkpoint。 &#x20;

    举个例子，当`save_total_limit=5`和指定`load_best_model_at_end`时，将始终保留最近的四个checkpoint以及最好的checkpoint；当`save_total_limit=1`和指定`load_best_model_at_end`时，会保存两个checkpoint：最后一个和最好的一个（如果它们不同一个）。 &#x20;
34. **`load_best_model_at_end `(bool, 可选, 默认为False)**：用于指定是否在训练结束时加载在训练过程中最好的checkpoint，设置为 True 时，就是帮你找到在验证集上指标最好的checkpoint并且保存，然后还会保存最后一个checkpoint，在普通的多epoch训练中，最好设置为True，但在大模型训练中，一般是一个epoch，使用的就是最后一个checkpoint。 &#x20;
35. **`save_safetensors`**\*\* (bool, 可选, 默认为False)\*\*：用于指定是否在保存和加载模型参数时使用 "safetensors"，"safetensors" 就是更好地处理了不同 PyTorch 版本之间的模型参数加载的兼容性问题。 &#x20;
36. **`save_on_each_node`**\*\* (bool, 可选, 默认为 False)\*\*：在进行多节点分布式训练时，是否在每个节点上保存checkpoint，还是仅在主节点上保存。注意如果多节点使用的是同一套存储设备，比如都是外挂的铜一个nas，开启后会报错，因为文件名称都一样。 &#x20;
37. **`use_cpu`**\*\* (bool, 可选, 默认为 False)\*\*：是否使用CPU训练。如果设置为False，将使用CUDA或其他可用设备。 &#x20;
38. **`seed`**\*\* (int, 可选, 默认为42)\*\*：用于指定训练过程的随机种子，可以确保训练的可重现性，主要用于model\_init，随机初始化权重参数。 &#x20;
39. **`data_seed`**\*\* (int, 可选)\*\*：用于指定数据采样的随机种子，如果没有设置将使用与seed相同的种子，可以确保数据采样的可重现性。 &#x20;
40. **`jit_mode_eval `(bool, 可选, 默认为False)**：用于指定是否在推理（inference）过程中使用 PyTorch 的 JIT（Just-In-Time）跟踪功能，PyTorch JIT 是 PyTorch 的一个功能，用于将模型的前向传播计算编译成高性能的机器代码，会加速模型的推理。 &#x20;
41. **`use_ipex `(bool, 可选, 默认为 False)**：用于指定是否使用英特尔扩展（Intel extension）来优化 PyTorch，需要安装IPEX，IPEX是一组用于优化深度学习框架的工具和库，可以提高训练和推理的性能，特别针对英特尔的处理器做了优化。 &#x20;
42. **`bf16 `(bool, 可选, 默认为False)**：用于指定是否使用bf16进行混合精度训练，而不是fp32训练，需要安培架构或者更高的NVIDIA架构。

    在简单解释一下混合精度训练：模型训练时将模型参数和梯度存储为fp32，但在前向和后向传播计算中使用fp16，这样可以减少内存使用和计算时间，并提高训练速度，这个只是简单的解释，关于混合精度训练，这篇文章讲的比较好 [点这里](https://mp.weixin.qq.com/s%3F__biz%3DMzI4MDYzNzg4Mw%3D%3D%26mid%3D2247550159%26idx%3D5%26sn%3Df5db2afa547970bc429112e32d2e7daf%26chksm%3Debb73c1bdcc0b50d0e85039bd5d8349a23330e3e0f138a7dd2da218a20174d0965837682dd14%26scene%3D27 "点这里")。 &#x20;
43. **`fp16 `(bool,** 可选, 默认为False)：用于指定是否使用fp16进行混合精度训练，而不是fp32训练。 &#x20;
44. **`fp16_opt_level `(str, 可选, 默认为 ''O1'')**：对于fp16训练，选择的Apex AMP的优化级别，可选值有 \['O0', 'O1', 'O2'和'O3']。详细信息可以看Apex文档。 &#x20;
45. **`half_precision_backend`**\*\* (str, 可选, 默认为"auto")\*\*：用于指定混合精度训练（Mixed Precision Training）时要使用的后端，必须是 "auto"、"cuda\_amp"、"apex"、"cpu\_amp" 中的一个。"auto"将根据检测到的PyTorch版本来使用后端，而其他选项将会强制使用请求的后端。使用默认就行。&#x20;
46. **`bf16_full_eval`**\*\* (bool, 可选, 默认为 False)\*\*：用于指定是否使用完全的bf16进行评估，而不是fp32。这样更快且省内存，但因为精度的问题指标可能会下降。 &#x20;
47. **`fp16_full_eval`**\*\* (bool, 可选, 默认为 False)\*\*：同上，不过将使用fp16. &#x20;
48. **`tf32`**\*\* (bool, 可选)\*\*：用于指定是否启用tf32精度模式，适用于安培架构或者更高的NVIDIA架构，默认值取决于PyTorch的版本torch.backends.cuda.matmul.allow\_tf32的默认值。 &#x20;
49. **`local_rank`**\*\* (int, 可选, 默认为 -1)\*\*：用于指定在分布式训练中的当前进程（本地排名）的排名，这个不需要我们设置，使用PyTorch分布式训练时会自动设置，默认为自动设置。 &#x20;
50. **`ddp_backend`**\*\* (str, 可选)\*\*：用于指定处理分布式计算的后端框架，这些框架的主要用于多个计算节点协同工作以加速训练，处理模型参数和梯度的同步、通信等操作，可选值如下 &#x20;
    - **"nccl"**：这是 NVIDIA Collective Communications Library (NCCL) 的后端。 &#x20;
    - **"mpi"**：Message Passing Interface (MPI) 后端， 是一种用于不同计算节点之间通信的标准协议。 &#x20;
    - **"ccl"**：这是 Intel的oneCCL (oneAPI Collective Communications Library) 的后端。 &#x20;
    - **"gloo"**：这是Facebook开发的分布式通信后端。 &#x20;
    - **"hccl"**：这是Huawei Collective Communications Library (HCCL) 的后端，用于华为昇腾NPU的系统上进行分布式训练。 &#x20;
      默认会根据系统自动设置，一般是nccl。 &#x20;
51. **`tpu_num_cores `(int, 可选)**：指定在TPU上训练时，TPU核心的数量。 &#x20;
52. **`dataloader_drop_last `(bool, 可选, 默认为False)**：用于指定是否丢弃最后一个不完整的batch，发生在数据集的样本数量不是batch\_size的整数倍的时候。 &#x20;
53. **`eval_steps `(int or float, 可选)**：如果evaluation\_strategy="steps"，就是指两次评估之间的更新步数，如果未设置，默认和设置和logging\_steps相同的值，如果是在\[0, 1)的浮点数，则就会当做与总评估步骤数的比例。 &#x20;
54. **`dataloader_num_workers `(int, 可选, 默认为 0)**：用于指定数据加载时的子进程数量（仅用于PyTorch）其实就是PyTorch的num\_workers参数，0表示数据将在主进程中加载。 &#x20;
55. **`past_index `(int, 可选, 默认为 -1)**：一些模型（如TransformerXL或XLNet）可以利用过去的隐藏状态进行预测，如果将此参数设置为正整数，Trainer将使用相应的输出（通常索引为2）作为过去状态，并将其在下一个训练步骤中作为mems关键字参数提供给模型，只针对一些特定模型。 &#x20;
56. **`run_name`**\*\* (str, 可选)\*\*：用于指定训练运行（run）的字符串参数，与日志记录工具（例如wandb和mlflow）一起使用，不影响训练过程，就是给其他的日志记录工具开了一个接口，个人还是比较推荐wandb比较好用。
57. **`disable_tqdm `(bool, 可选)**：是否禁用Jupyter笔记本中的\~notebook.NotebookTrainingTracker生成的tqdm进度条，如果日志级别设置为warn或更低，则将默认为True，否则为False。 &#x20;
58. **`remove_unused_columns `(bool, 可选, 默认为True)**：是否自动删除模型在训练时，没有用到的数据列，默认会删除，比如你的数据有两列分别是content和id，如果没有用到id这一列，训练时就会被删除。 &#x20;
59. **`label_names `(List\[str], 可选)**：用于指定在模型的输入字典中对应于标签（labels）的键，默认情况下不需要显式指定。 &#x20;
60. **`metric_for_best_model`**\*\* (str, 可选)\*\*：与 load\_best\_model\_at\_end 结合使用，用于指定比较不同模型的度量标准，默认情况下，如果未指定，将使用验证集的 "loss" 作为度量标准，可使用accuracy、F1、loss等。 &#x20;
61. **`greater_is_better `(bool, 可选)**：与 load\_best\_model\_at\_end 和 metric\_for\_best\_model 结合使用，这个和上面的那个参数是对应的，是指上面的那个指标是越大越好还是越小越好，如果是loss就是越小越好，这个参数就会被设置为False；如果是accuracy，你需要把这个值设为True。 &#x20;
62. **`ignore_data_skip `(bool, 可选，默认为False)**：用于指定是否断点训练，即训练终止又恢复后，是否跳过之前的训练数据。 &#x20;
63. **`resume_from_checkpoint `(str, 可选)**：用于指定从checkpoint恢复训练的路径。 &#x20;
64. **`sharded_ddp `(bool, str 或 ShardedDDPOption 列表, 可选, 默认为'')**：是否在分布式训练中使用 Sharded DDP（Sharded Data Parallelism），这是由 FairScale提供的，默认不使用，简单解释一下： FairScale 是Mate开发的一个用于高性能和大规模训练的 PyTorch 扩展库。这个库扩展了基本的 PyTorch 功能，同时引入了最新的先进规模化技术，通过可组合的模块和易于使用的API，提供了最新的分布式训练技术。详细的可以看其官网。 &#x20;
65. **`fsdp `(bool, str 或 FSDPOption 列表, 可选, 默认为'')**：用于指定是否要启用 PyTorch 的 FSDP（Fully Sharded Data Parallel Training），以及如何配置分布式并行训练。 &#x20;
66. **`fsdp_config `(str 或 dict, 可选)**：用于配置 PyTorch 的 FSDP（Fully Sharded Data Parallel Training）的配置文件 &#x20;
67. **`deepspeed `(str 或 dict, 可选)**：用于指定是否要启用 DeepSpeed，以及如何配置 DeepSpeed。也是目前分布式训练使用最多的框架，比上面pytorch原生分布式训练以及FairScale用的范围更广，详细的可以看其官网。 &#x20;
68. **`label_smoothing_factor`**\*\* (float, 可选，默认为0.0)\*\*：用于指定标签平滑的因子。 &#x20;
69. **`debug`**\*\* (str 或 DebugOption 列表, 可选, 默认为'')\*\*：用于启用一个或多个调试功能 &#x20;

    支持的选项： &#x20;
    - "underflow\_overflow"：此选项用于检测模型输入/输出中的溢出。 &#x20;
    - "tpu\_metrics\_debug"：此选项用于在 TPU 上打印调试指标。
70. **`optim`**\*\* (str 或 training\_args.OptimizerNames, 可选, 默认为 "adamw\_torch")\*\*：指定要使用的优化器。 &#x20;

    可选项： &#x20;
    - "adamw\_hf" &#x20;
    - "adamw\_torch" &#x20;
    - "adamw\_torch\_fused" &#x20;
    - "adamw\_apex\_fused" &#x20;
    - "adamw\_anyprecision" &#x20;
    - "adafactor"
71. **`optim_args`**\*\* (str, 可选)\*\*：用于向特定类型的优化器（如adamw\_anyprecision）提供额外的参数或自定义配置。 &#x20;
72. **`group_by_length`**\*\* (bool, 可选, 默认为 False)\*\*：是否在训练数据集中对大致相同长度的样本进行分组然后放在一个batch里，目的是尽量减少在训练过程中进行的padding，提高训练效率。 &#x20;
73. **`length_column_name`**\*\* (str, 可选, 默认为 "length")\*\*：当你上个参数设置为True时，你可以给你的训练数据在增加一列”长度“，就是事先计算好的，可以加快分组的速度，默认是length。 &#x20;
74. **`report_to`**\*\* (str 或 str 列表, 可选, 默认为 "all")\*\*：用于指定要将训练结果和日志报告到的不同日记集成平台，有很多"azure\_ml", "clearml", "codecarbon", "comet\_ml", "dagshub", "flyte", "mlflow", "neptune", "tensorboard", and "wandb"。直接默认就行，都发。 &#x20;
75. **`ddp_find_unused_parameters`**\*\* (bool, 可选)\*\*：当你使用分布式训练时，这个参数用于控制是否查找并处理那些在计算中没有被使用的参数，如果启用了梯度检查点（gradient checkpointing），表示部分参数是惰性加载的，这时默认值为 False，因为梯度检查点本身已经考虑了未使用的参数，如果没有启用梯度检查点，默认值为 True，表示要查找并处理所有参数，以确保它们的梯度被正确传播。 &#x20;
76. **`ddp_bucket_cap_mb`**\*\* (int, 可选)\*\*：在分布式训练中，数据通常分成小块进行处理，这些小块称为"桶"，这个参数用于指定每个桶的最大内存占用大小，一般自动分配即可。 &#x20;
77. **`ddp_broadcast_buffers`**\*\* (bool, 可选)\*\*：在分布式训练中，模型的某些部分可能包含缓冲区，如 Batch Normalization 层的统计信息，这个参数用于控制是否将这些缓冲区广播到所有计算设备，以确保模型在不同设备上保持同步，如果启用了梯度检查点，表示不需要广播缓冲区，因为它们不会被使用，如果没有启用梯度检查点，默认值为 True，表示要广播缓冲区，以确保模型的不同部分在所有设备上都一致。 &#x20;
78. **`gradient_checkpointing`**\*\* (bool, 可选, 默认为False)\*\*：是否开启梯度检查点，简单解释一下：训练大型模型时需要大量的内存，其中在反向传播过程中，需要保存前向传播的中间计算结果以计算梯度，但是这些中间结果占用大量内存，可能会导致内存不足，梯度检查点会在训练期间释放不再需要的中间结果以减小内存占用，但它会使训练变慢。 &#x20;
79. **`dataloader_pin_memory`**\*\* (bool, 可选, 默认为 True)\*\*：用于指定dataloader加载数据时，是否启用“pin memory”功能。“Pin memory” 用于将数据加载到GPU内存之前，将数据复制到GPU的锁页内存（pinned memory）中，锁页内存是一种特殊的内存，可以更快地传输数据到GPU，从而加速训练过程，但是会占用额外的CPU内存，会导致内存不足的问题，如果数据量特别大，百G以上建议False。 &#x20;
80. **`skip_memory_metrics`**\*\* (bool, 可选, 默认为 True)\*\*：用于控制是否将内存分析报告添加到性能指标中，默认情况下跳过这一步，以提高训练和评估的速度，建议打开，更能够清晰的知道每一步的内存使用。 &#x20;
81. **`include_inputs_for_metrics`**\*\* (bool, 可选, 默认为 False)\*\*：是否将输入传递给 `compute_metrics` 函数，一般计算metrics用的是用的是模型预测的结果和我们提供的标签，但是有的指标需要输入，比如cv的IoU（Intersection over Union）指标。 &#x20;
82. **`auto_find_batch_size`**\*\* (bool, 可选, 默认为 False)\*\*：是否使用自动寻找适合内存的batch size大小，以避免 CUDA 内存溢出错误，需要安装 `accelerate`（使用 `pip install accelerate`），这个功能还是比较NB的。 &#x20;
83. **`full_determinism`**\*\* (bool, 可选, 默认为 False)\*\*：如果设置为 `True`，将调用 `enable_full_determinism()` 而不是 `set_seed()`，训练过程将启用完全确定性（full determinism），在训练过程中，所有的随机性因素都将被消除，确保每次运行训练过程都会得到相同的结果，注意：会对性能产生负面影响，因此仅在调试时使用。 &#x20;
84. **`torchdynamo`**\*\* (str, 可选)\*\*：用于选择 TorchDynamo 的后端编译器，TorchDynamo 是 PyTorch 的一个库，用于提高模型性能和部署效率，可选的选择包括 "eager"、"aot\_eager"、"inductor"、"nvfuser"、"aot\_nvfuser"、"aot\_cudagraphs"、"ofi"、"fx2trt"、"onnxrt" 和 "ipex"。默认就行，自动会选。 &#x20;
85. **`ray_scope`**\*\* (str, 可选, 默认为 "last")\*\*：用于使用 Ray 进行超参数搜索时，指定要使用的范围，默认情况下，使用 "last"，Ray 将使用所有试验的最后一个检查点，比较它们并选择最佳的。详细的可以看一下它的文档。 &#x20;
86. **`ddp_timeout`**\*\* (int, 可选, 默认为 1800)\*\*：用于 torch.distributed.init\_process\_group 调用的超时时间，在分布式运行中执行较慢操作时，用于避免超时，具体的可以看 [PyTorch 文档](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/distributed.html%23torch.distributed.init_process_group "PyTorch 文档") 。 &#x20;
87. **`torch_compile`**\*\* (bool, 可选, 默认为 False)\*\*：是否使用 PyTorch 2.0 及以上的 torch.compile 编译模型，具体的可以看 [PyTorch 文档](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/distributed.html%23torch.distributed.init_process_group "PyTorch 文档") 。 &#x20;
88. **`torch_compile_backend`**\*\* (str, 可选)\*\*：指定在 torch.compile 中使用的后端，如果设置为任何值，将启用 torch\_compile。 &#x20;
89. **`torch_compile_mode`**\*\* (str, 可选)\*\*：指定在 torch.compile 中使用的模式，如果设置为任何值，将启用 torch\_compile。 &#x20;
90. **`include_tokens_per_second`**\*\* (bool, 可选)\*\*：确定是否计算每个设备的每秒token数以获取训练速度指标，会在整个训练数据加载器之前进行迭代，会稍微减慢整个训练过程，建议打开。 &#x20;
91. **`push_to_hub`**\*\* (bool, 可选, 默认为 False)\*\*：指定是否在每次保存模型时将模型推送到Huggingface Hub。 &#x20;
92. **`hub_model_id`**\*\* (str, 可选)\*\*：指定要与本地 output\_dir 同步的存储库的名称。 &#x20;
93. **`hub_strategy`**\*\* (str 或 HubStrategy, 可选, 默认为 "every\_save") \*\*：指定怎么推送到Huggingface Hub。
94. **`hub_token`**\*\* (str, 可选)\*\*：指定推送模型到Huggingface Hub 的token。 &#x20;
95. **`hub_private_repo`**\*\* (bool, 可选, 默认为 False)\*\*：如果设置为 True，Huggingface Hub 存储库将设置为私有。 &#x20;
96. **`hub_always_push`**\*\* (bool, 可选, 默认为 False)\*\*：是否每次都推送模型。 &#x20;
