 [**English**](README.md) | [**中文说明**](README_ZH.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="500"/>
    <br>
<p>
<p>
<p align="center">
    <a href="https://github.com/airaria/TextPruner/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/airaria/TextPruner.svg?color=green&style=flat-square">
    </a>
    <a href="https://TextPruner.readthedocs.io/">
        <img alt="Documentation" src="https://img.shields.io/website?down_message=offline&label=Documentation&up_message=online&url=https%3A%2F%2FTextPruner.readthedocs.io">
    </a>    
    <a href="https://pypi.org/project/TextPruner">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/TextPruner">
    </a>    
    <a href="https://github.com/airaria/TextPruner/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/airaria/TextPruner?include_prereleases">
    </a>
</p>

**TextPruner**是一个为预训练语言模型设计的模型裁剪工具包，通过轻量、快速的裁剪方法对模型进行结构化剪枝，从而实现压缩模型体积、提升模型速度。

其他相关资源：

- 知识蒸馏工具TextBrewer：https://github.com/airaria/TextBrewer
- 中文MacBERT预训练模型：https://github.com/ymcui/MacBERT
- 中文ELECTRA预训练模型：https://github.com/ymcui/Chinese-ELECTRA
- 中文XLNet预训练模型：https://github.com/ymcui/Chinese-XLNet
- 少数民族语言预训练模型CINO：https://github.com/ymcui/Chinese-Minority-PLM


## 目录

<!-- TOC -->

| 章节 | 内容 |
|-|-|
| [简介](#简介) | TextPruner介绍 |
| [安装](#安装) | 安装要求与方法 |
| [裁剪模式](#裁剪模式) | 三种裁剪模式说明 |
| [使用方法](#使用方法) | TextPruner快速上手 |
| [实验结果](#实验结果) | 典型任务上的裁剪效果 |
| [常见问题](#常见问题) | 常见问题 |
| [关注我们](#关注我们) | - |

## 简介

**TextPruner**是一个为预训练语言模型设计，基于PyTorch实现的模型裁剪工具包。它提供了针对预训练模型的结构化裁剪功能，通过识别并移除模型结构中不重要的结构与神经元，达到压缩模型大小、提升模型推理速度的目的。

TextPruner的主要特点包括：

* **功能通用**: TextPruner适配多种预训练模型，并适用于多种NLU任务。除了标准预训练模型外，用户也可使用TextPruner裁剪基于标准预训练模型开发的自定义模型
* **灵活便捷**: TextPruner即可作为Python包在Python脚本中使用，也提供了单独命令行工具。
* **运行高效**: TextPruner使用无训练的结构化裁剪方法，运行迅速，快于知识蒸馏等基于训练的方法。

TextPruner目前支持词表裁剪和transformer裁剪，参见[裁剪模式](#裁剪模式)。

要使用TextPruner，用户可以在python脚本中导入TextPruner或直接在命令行运行TextPruner命令行工具，参见[使用方法](#使用方法)。

TextPruner在典型任务上的实验效果，参见[实验结果](#实验)。

TextPruner目前支持[Transformers](https://github.com/huggingface/transformers)库中的如下预训练模型:
* BERT
* Albert
* Electra
* RoBERTa
* XLM-RoBERTa

API文档参见[在线文档](https://textpruner.readthedocs.io)

## 安装

* 安装要求

    * Python >= 3.7
    * torch >= 1.7
    * transformers >= 4.0
    * sentencepiece
    * protobuf

* 使用pip安装

    ```bash
    pip install textpruner
    ```

*  从源代码安装

    ```bash
    git clone https://github.com/airaria/TextPruner.git
    pip install ./textpruner
    ```

### 裁剪模式

TextPruner提供了3种裁剪模式，分别为**词表裁剪（Vocabulary Pruning）**，**Transformer裁剪（Transformer Pruning）**和**流水线裁剪（Pipeline Pruning）**。


![](pics/PruningModes.png)

#### 词表裁剪

预训练模型通常包含对具体任务来说冗余的词表。通过移除词表中未在具体任务未出现的token，可以实现减小模型体积，提升MLM等任务训练速度的效果。

#### Transformer裁剪

另一种裁剪方式是裁剪每个transformer模块的大小。一些研究表明transformer中的注意力头（attention heads）并不是同等重要，移除不重要的注意力头并不会显著降低模型性能。TextPruner找到并移除每个transformer中“不重要”的注意力头和全连接层神经元，从而在减小模型体积的同时把对模型性能的影响尽可能降到最低。

#### 流水线裁剪

在该模式中，TextPruner对给定模型依次分别进行Transformer裁剪和词表裁剪，对模型体积做全面的压缩。


## 使用方法

**Pruners**执行具体的裁剪过程，**configurations**设置裁剪参数。它们名称的含义是不言自明的：
* Pruners
  * `textpruner.VocabularyPruner`
  * `textpruner.TransformerPruner`
  * `textpruner.PipelinePruner`
* Configurations
  * `textpruner.GeneralConfig`
  * `textpruner.VocabularyPruningConfig`
  * `textpruner.TransformerPruningConfig`


TextPruner的API文档请参见[在线文档](https://textpruner.readthedocs.io)。
Configurations的说明参见[Configurations](#configurations)。
下面展示基本用法。

### 词表裁剪

要进行词表裁剪，用户应提供一个文本文件或字符串列表（list of strings）。TextPruner将从model和tokenizer中移除未在文本文件或列表中出现过的token。

具体的例子参见[examples/vocabulary_pruning](examples/vocabulary_pruning)和[examples/vocabulary_pruning_xnli](examples/vocabulary_pruning_xnli).

#### 在脚本中使用

词表裁剪仅需3行代码：

```python
from textpruner import VocabularyPruner
pruner = VocabularyPruner(model, tokenizer)
pruner.prune(dataiter=texts)
```

* `model`和`tokenizer`是要裁剪的模型和对应的分词器
* `texts`是字符串列表（list of strings），一般为任务相关数据的文本，用以确定裁剪后的词表大小。TextPruner将从model和tokenizer中移除未在其中出现过的token。



#### 使用命令行工具

```bash
textpruner-cli  \
  --pruning_mode vocabulary \
  --configurations gc.json vc.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path /path/to/model/and/config/directory \
  --vocabulary /path/to/a/text/file
```
* `configurations`：JSON格式的配置文件。
* `model_class` : 模型的完整类名，要求该类在当前目录下可访问。例如`model_class`是`modeling.ModelClassName`，那么当前目录下应存在`modeling.py`。如果`model_class` 中无模块名，那么TextPruner会试图从transformers库中导入`model_class`，如上面的例子。
* `tokenizer_class` : tokenizer的完整类名。要求该类在当前目录下可访问。如果`tokenizer_class` 中无模块名，那么TextPruner会试图从transformers库中导入`tokenizer_class`。
* `model_path`：模型、tokenizer和相关配置文件存放目录。
* `vocabulary` : 用于定义新词表的文本文件。TextPruner将从model和tokenizer中移除未在其中出现过的token。


### Transformer裁剪

* 要在一个数据集上进行transformer裁剪，需要一个`dataloader`对象。每次迭代`dataloader`应返回一个batch，batch的格式应与训练模型时相同：包括inputs和labels（batch内容本身不必用和训练时相同）。
* TextPruner需要模型返回的loss用以计算神经元的重要性指标。TextPruner会尝试猜测模型输出中的哪个元素是loss。**如以下皆不成立**：
  * 模型只返回一个元素，那个元素就是loss
  * 模型返回一个list或tuple。loss是其中第一个元素
  * loss可以通过`output['loss']`或`output.loss`得到，其中`output`是模型的输出
  
  那么用户应提供一个`adaptor`函数（以模型的输出为输入，返回loss）给`TransformerPruner`。
* 当运行于自监督裁剪模式，TextPruner需要模型返回的logits。此时需要`adaptor`函数返回logits。详细参见`TransformerPruningConfig`中的`use_logits`选项。

具体的例子参见[examples/transformer_pruning](examples/transformer_pruning)

自监督裁剪的例子参见[examples/transformer_pruning_xnli](examples/transformer_pruning_xnli)

#### 在脚本中使用

裁剪一个12层预训练模型，每层的注意力头目标数为8，全连接层的目标维数为2048，通过4次迭代裁剪到目标大小：
```python
from textpruner import TransformerPruner, TransformerPruningConfig
transformer_pruning_config = TransformerPruningConfig(
      target_ffn_size=2048, 
      target_num_of_heads=8, 
      pruning_method='iterative',
      n_iters=4)
pruner = TransformerPruner(model,transformer_pruning_config=transformer_pruning_config)   
pruner.prune(dataloader=dataloader, save_model=True)
```

* transformer_pruning_config设置了具体的裁剪参数。
* `dataloader`用于向pruner提供数据用于计算各个注意力头的神经元的重要性，从而决定裁剪顺序。

#### 使用命令行工具

```bash
textpruner-cli  \
  --pruning_mode transformer \
  --configurations gc.json tc.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path ../models/xlmr_pawsx \
  --dataloader_and_adaptor dataloader_script
```
* `configurations`：JSON格式的配置文件。
* `model_class` : 模型的完整类名，要求该类在当前目录下可访问。例如`model_class`是`modeling.ModelClassName`，那么当前目录下应存在`modeling.py`。如果`model_class` 中无模块名，那么TextPruner会试图从transformers库中导入`model_class`，如上面的例子。
* `tokenizer_class` : tokenizer的完整类名。要求该类在当前目录下可访问。如果`tokenizer_class` 中无模块名，那么TextPruner会试图从transformers库中导入`tokenizer_class`。
* `model_path`：模型、tokenizer和相关配置文件存放目录。
* `dataloader_and_adaptor` : Python脚本文件，其中定义并初始化了dataloader和adaptor（adaptor可选）。



### 流水线裁剪

流水线裁剪结合了transformer裁剪和词表裁剪。

具体的例子参见[examples/pipeline_pruning](examples/pipeline_pruning)

#### 在脚本中使用

```python
from textpruner import PipelinePruner, TransformerPruningConfig
transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=2048, target_num_of_heads=8, 
    pruning_method='iterative',n_iters=4)
pruner = PipelinePruner(model, tokenizer, transformer_pruning_config=transformer_pruning_config)
pruner.prune(dataloader=dataloader, dataiter=texts, save_model=True)
```

#### 使用命令行工具

```bash
textpruner-cli  \
  --pruning_mode pipeline \
  --configurations gc.json tc.json vc.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path ../models/xlmr_pawsx \
  --vocabulary /path/to/a/text/file \
  --dataloader_and_adaptor dataloader_script
```

### Configurations

裁剪过程受配置对象（configuration objects）控制：

* `GeneralConfig`：设置使用的device和输出目录。
* `VocabularyPruningConfig`：设置裁剪的阈值（token的词频低于此阈值将被裁减），以及是否裁剪`lm_head`。
* `TransformerPruningConfig`：Transformer裁剪过程参数的各种配置。

它们用于不同的裁剪模式：

* 词表裁剪可接受`GeneralConfig` and `VocabularyPruningConfig`

  ```python
  VocabularyPruner(vocabulary_pruning_config= ..., general_config = ...)
  ```

* Transformer裁剪可接受`GeneralConfig` and `TransformerPruningConfig`
  ```python
  TransformerPruner(transformer_pruning_config= ..., general_config = ...)
  ```

* 流水线裁剪可接受全部3种Config：
  ```python
  TransformerPruner(transformer_pruning_config= ..., vocabulary_pruning_config= ..., general_config = ...)
  ```

在Python脚本中，配置对象是dataclass对象；在命令行中，配置对象是JSON文件。
如果未向pruner提供相应的配置对象，TextPruner将使用默认配置。
配置对象的各个参数详细意义请参见`GeneralConfig`，`VocabularyPruningConfig`和`TransformerPruningConfig` API文档。


在Python脚本定义：

```python
from textpruner import GeneralConfig, VocabularyPruningConfig, TransformerPruningConfig
from textpruner import VocabularyPruner, TransformerPruner, PipelinePruner

#GeneralConfig
general_config = GeneralConfig(device='auto',output_dir='./pruned_models')

#VocabularyPruningConfig
vocabulary_pruning_config = VocabularyPruningConfig(min_count=1,prune_lm_head='auto')

#TransformerPruningConfig
#Pruning with the given masks 
transformer_pruning_config = TransformerPruningConfig(pruning_method = 'masks')

#TransformerPruningConfig
#Pruning on labeled dataset iteratively
transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size  = 2048,
    target_num_of_heads = 8,
    pruning_method = 'iterative',
    ffn_even_masking = True,
    head_even_masking = True,
    n_iters = 1,
    multiple_of = 1
)
```

作为JSON文件：

* `GeneralConfig`：[gc.json](examples/configurations/gc.json)
* `VocabularyPruningConfig`：[vc.json](examples/configurations/vc.json)
* `TransformerPruningConfig`：
    * 使用给定的masks进行裁剪：[tc-masks.json](examples/configurations/tc-masks.json)
    * 在给定数据集上迭代裁剪：[tc-iterative.json](examples/configurations/tc-iterative.json)

### 辅助函数

* `textpruner.summary`：显示模型参数摘要。
* `textpruner.inference_time`：测量与显示模型的推理耗时。

例子：

```python
from transformers import BertForMaskedLM
import textpruner
import torch

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print("Model summary:")
print(textpruner.summary(model,max_level=3))

dummy_inputs = [torch.randint(low=0,high=10000,size=(32,512))]
print("Inference time:")
textpruner.inference_time(model.to('cuda'),dummy_inputs)
```

Outputs:

```
Model summary:
LAYER NAME                          	        #PARAMS	     RATIO	 MEM(MB)
--model:                            	    109,514,810	   100.00%	  417.77
  --bert:                           	    108,892,160	    99.43%	  415.39
    --embeddings:                   	     23,837,696	    21.77%	   90.94
      --position_ids:               	            512	     0.00%	    0.00
      --word_embeddings:            	     23,440,896	    21.40%	   89.42
      --position_embeddings:        	        393,216	     0.36%	    1.50
      --token_type_embeddings:      	          1,536	     0.00%	    0.01
      --LayerNorm:                  	          1,536	     0.00%	    0.01
    --encoder
      --layer:                      	     85,054,464	    77.66%	  324.46
  --cls
    --predictions(partially shared):	        622,650	     0.57%	    2.38
      --bias:                       	         30,522	     0.03%	    0.12
      --transform:                  	        592,128	     0.54%	    2.26
      --decoder(shared):            	              0	     0.00%	    0.00

Inference time:
Device: cuda:0
Mean inference time: 1214.41ms
Standard deviation: 2.39ms
```


## 实验结果

使用基于[XLM-RoBERTa-base](https://github.com/facebookresearch/XLM)的分类模型，我们在多语言NLI任务[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx)的英文数据集上训练与测试，并使用TextPruner对训练好的模型进行裁剪。

### 词表裁剪

我们从[XNLI英文训练集](https://github.com/facebookresearch/XNLI)中采样[10万条样本](examples/datasets/xnli/en.tsv)作为词表，将XLM-RoBERTa模型的词表大小裁剪至这10万条样本的范围内，裁剪前后模型对比如下所示。

| Model                | Total size (MB) | Vocab size | Acc on en (%)|
| :-------------------- | :---------------: | :----------: | :------------: |
| XLM-RoBERTa-base     | 1060 (100%)     | 250002     | 94.65        |
| + Vocabulary Pruning | 398 (37.5%)     | 23936      | 94.20        |

XLM-RoBERTa作为多语言模型，词表占了模型的很大一部分。通过只保留相关任务和语言的词表，可以显著减小模型体积,并且对模型准确率只有微弱影响。

### Transfomer裁剪

使用（H,F）指示模型结构，其中H是平均每层注意力头数量，F是全连接层的维数（intermediate hidden size）。原始的XLM-RoBERTa-base模型可记为（12, 3072）。我们考虑裁剪到另外两种结构（8,2048）和（6，1536）。

#### 推理时间

使用长度512，batch size 32的数据作为输入测量推理时间：

| Model      | Total size (MB) | Encoder size (MB) | Inference time (ms) | Speed up |
| :---------- | :---------------: | :-----------------: | :-------------------: | :--------: |
| (12, 3072) | 1060            | 324               | 1012                | 1.0x     |
| (8, 2048)  | 952             | 216               | 666                 | 1.5x     |
| (6, 1536)  | 899             | 162               | 504                 | 2.0x     |


#### 任务性能

我们尝试使用不同的迭代次数进行transformer裁剪，各个模型的准确率变化如下表所示：

| Model      | n_iters=1           |           n_iters=2 |           n_iters=4 |           n_iters=8 |           n_iters=16 |
| :------------ | :-----------: | :-----------: | :-----------: | :-----------: | :------------: |
| (12, 3072)   | 94.65       | -           | -           | -           | -            |
| (8, 2048)    | 93.30       | 93.60       | 93.60       | 93.85       | 93.95        |
| (8, 2048) with uneven heads   | 92.95       | 93.50       | 93.95       | 94.05        | **94.25**    |
| (6, 1536)    | 85.15       | 89.10       | 90.90       | 90.60       | 90.85        |
| (6, 1536) with uneven heads   | 45.35       | 86.45       |  90.55     | 90.90         | **91.95**    |

表中的uneven heads指允许模型在不同层有不同的注意力头数。
可以看到，随着迭代次数的增加，裁剪后的模型的性能也随之提升。

### 流水线裁剪

最后，我们用PipelinePruner同时裁剪词表和transformer：

| Model                                             | Total size (MB) | Speed up | Acc on en (%) |
| :-----------------------------------------------  | :-------------: | -------- | ------------- |
| XLM-RoBERTa-base                                  |   1060 (100%)   | 1.0x     | 94.65         |
| + Pipeline pruning to (8, 2048) with uneven heads |    227 (22%)    | 1.5x     | 93.75         |

Transformer裁剪过程使用了16次迭代，词表裁剪过程使用XNLI英文训练集中采样的10万条样本作为词表。整个裁剪过程在单张T4 GPU上耗时10分钟。

## 常见问题

**Q: TextPruner 是否支持 Tensorflow 2 ？**

A: 不支持。

**Q: 对于知识蒸馏和模型裁剪，能否给一些使用建议 ？**

A: 知识蒸馏与模型裁剪都是减小模型体积的主流手段：

* 知识蒸馏通常可以获得更好的模型效果和更高的压缩率，但是蒸馏过程较消耗算力与时间；为了获得好的蒸馏效果，对大量数据的访问也是必不可少的。

* 在相同目标模型体积下，结构化无训练裁剪方法的性能通常低于知识蒸馏，但其优点是快速与轻量。裁剪过程最短可以在数分钟内完成，并且只需要少量标注数据进行指导。

（还有一类包含训练过程的裁剪方法，它们在保证模型性能的同时也可以取得很好的压缩效果。）

如果你对知识蒸馏感兴趣，可以参见我们的知识蒸馏工具包[TextBrewer](http://textbrewer.hfl-rc.com)。

如果你想取得最好的模型压缩效果，或许可以尝试同时采用蒸馏与裁剪这两种手段。

## 关注我们

欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态

![](pics/hfl_qrcode.jpg)
