# 项目名称: LLM-SFT

## 项目简介
本项目旨在通过LoRA（Low-Rank Adaptation）技术对不同的语言模型进行微调。项目包含数据集、模型、输出、设置和工具等文件夹。

## 目录结构

- **datasets**: 存放训练和测试数据集。
- **models**: 保存训练好的模型权重文件。
- **outputs**: 存放训练输出结果，包括日志和生成的模型。
- **settings**: 配置文件夹，包含不同训练任务的JSON配置文件。
  - `lora_llama_course1.json`: 用于第一个LLaMA模型微调的配置文件。
  - `lora_llama_course2.json`: 用于第二个LLaMA模型微调的配置文件。
  - `lora_qwen_course1.json`: 用于QWEN模型微调的配置文件。
- **utils**: 工具文件夹，包含数据处理和辅助脚本。
  - `data_preprocess.py`: 数据处理工具脚本。
  - `news_collecting.py`: 新闻收集脚本
- **inference_example.ipynb**: 推理示例Jupyter Notebook。
- **README.md**: 项目说明文件。
- **train.py**: 主训练脚本。

## 使用方法

### 准备环境
1. 克隆项目到本地并让终端进入该文件夹

2. 安装所需依赖:

   环境说明：
   - python: 3.10
   - cuda: 11.3
   - pytorch: 1.12.1
   - bitsandbytes: 0.39

   以下安装步骤仅为cuda<11.4特化。对于更高版本无需如此，请直接参考hugging face官网。
   ```bash
    conda create -n sft python=3.10 notebook
    activate sft
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
    pip install transformers accelerate trl bitsandbytes==0.39 scipy deepspeed hf_transfer modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install transformers_stream_generator tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install peft --no-dependencies -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   Note: 需要把bitsandbytes库中的__init__文件中if("usr/local")块注释掉

### 数据与模型准备

#### 下载数据集

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --repo-type dataset --resume-download Maciel/FinCUGE-Instruction  --local-dir data --local-dir-use-symlinks False
huggingface-cli download --repo-type dataset --resume-download silk-road/alpaca-data-gpt4-chinese  --local-dir data --local-dir-use-symlinks False
```

#### 下载模型

魔搭社区上查找模型路径，示例如下：
- 'LLM-Research/Meta-Llama-3-8B'
- "FlagAlpha/Llama3-Chinese-8B-Instruct"  
- "ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora"  
- 'zhuangxialie/Llama3_Chinese_Sft'
- 'qwen/Qwen-7B-Chat'

```bash
python
from modelscope import snapshot_download
model_id = 'qwen/Qwen-7B-Chat'          
model_dir = snapshot_download(model_id, cache_dir='./models/')
quit()
```

### 模型训练
根据需要修改 `settings` 文件夹中的配置文件，然后运行 `train.py` 进行模型训练:
```bash
python train.py --config settings/lora_llama_course1.json
```

### 模型推理
请参考 `inference_example.ipynb` 文件，按照里面的步骤进行模型推理。

