import json
import argparse
import torch
import transformers
from typing import Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, EvalPrediction
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel,PeftConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_pt_utils import LabelSmoother

from utils import data_to_tokenized_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
    
# 加载模型和分词器
def load_model_and_tokenizer(config):
    assert "model_id" in config, "model_id not set"
    assert "lora_path" in config or "lora_config" in config, "lora path or lora config not set"
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], device_map="auto", torch_dtype=torch.float16, load_in_8bit=True, trust_remote_code=True)

    # 加载lora权重
    if "lora_path" in config:
        old_lora_config = PeftConfig.from_pretrained(config["lora_path"])
        model = PeftModel.from_pretrained(model, model_id=config["lora_path"],config=old_lora_config, is_trainable= True) 
    else:
        model = get_peft_model(model, LoraConfig(**config["lora_config"]))
    return tokenizer, model


def load_datasets(dataset_settings, sample_size):
    assert dataset_settings, "dataset_settings is not set"

    weights = 0
    for item in dataset_settings:
        weights += item["weight"] if "weight" in item else 1
    # 加载数据集并按权重抽取样本
    samples = []
    for item in dataset_settings:
        path, weight = item["path"], item["weight"] if item["weight"] else 1
        dataset = load_dataset(path, split='train')  # 只关心第一个split
        if "rename_mapping" in item:
            dataset = dataset.select_columns(list(item["rename_mapping"].keys()))
            for old_name, new_name in item["rename_mapping"].items():
                dataset = dataset.rename_column(old_name, new_name) 
        if "filter_cols" in item:
            dataset = dataset.filter(lambda example: example['task'] in item["filter_cols"])
        num_samples = min(len(dataset), sample_size)
        proportion = weight / weights  # 计算当前数据集应该抽取的比例
        samples.append(dataset.select(range(int(proportion * num_samples))))  # 抽取样本

    # 合并样本
    combined_dataset = concatenate_datasets(samples)

    # 打乱数据集
    combined_dataset = combined_dataset.shuffle(seed=42)
    print(combined_dataset)

    # 划分训练集和测试集
    train_test_split = combined_dataset.train_test_split(test_size=0.3)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    return train_dataset,test_dataset




# 开销信息函数
def get_gpu_memory_info():
    # 获取GPU数量
    num_gpus = torch.cuda.device_count()

    # 初始化总内存和总保留内存
    total_start_gpu_memory = 0
    total_used_memory = 0

    # 遍历所有GPU
    for i in range(num_gpus):
        total_start_gpu_memory += torch.cuda.max_memory_reserved(i)
        total_used_memory += torch.cuda.max_memory_allocated(i)

    # 将字节转换为GB
    total_start_gpu_memory_gb = round(total_start_gpu_memory / 1024 / 1024 / 1024, 3)
    total_used_memory_gb = round(total_used_memory / 1024 / 1024 / 1024, 3)

    # 计算使用百分比
    used_percentage = round(total_used_memory / total_start_gpu_memory * 100, 3)

    # 返回结果
    return {
        "num_gpus": num_gpus,
        "total_start_gpu_memory_gb": total_start_gpu_memory_gb,
        "total_used_memory_gb": total_used_memory_gb,
        "used_percentage": used_percentage
    }



def main(config,resume):

    tokenizer, model = load_model_and_tokenizer(config)
    train_dataset,test_dataset = load_datasets(config["dataset_settings"], config["sample_size"])

    # preprocess dataset
    if config['base']=='qwen':
        tokenizer.pad_token_id = tokenizer.eod_id
        train_dataset_tokenized = data_preprocess.preprocess_data_for_qwen(sources = train_dataset, tokenizer = tokenizer, max_len = config["max_len"] )
    elif config['base']=='llama':
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset_tokenized = data_preprocess.preprocess_data_for_llama(sources = train_dataset, tokenizer = tokenizer, max_len = 512 )
    else:
        raise NotImplementedError("have not chose a base yet or not support for it.")

    if  "gradient_checkpointing" in config["training_args"] and config["training_args"]["gradient_checkpointing"]==True:
        model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config["training_args"]),
        train_dataset=train_dataset_tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print(f"resume from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    gpu_memory_info = get_gpu_memory_info()
    print(f"Total GPUs: {gpu_memory_info['num_gpus']}")
    print(f"Total peak reserved memory across all GPUs: {gpu_memory_info['total_start_gpu_memory_gb']} GB.")

    # 保存lora增量模型
    model.save_pretrained(config["new_lora_path"] if "new_lora_path" in config else config["lora_path"])
    tokenizer.save_pretrained(config["new_lora_path"] if "new_lora_path" in config else config["lora_path"])


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process some JSON configurations.")
    parser.add_argument('--config_file', type=str, help="The JSON configuration file to load.")  
    parser.add_argument('--resume', help="resume from checkpoint", action="store_true")
    args = parser.parse_args()
    print(f"Selected config files: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = defaultdict(lambda: None, json.load(f))
    main(config,args.resume)

# use case:
#  CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file settings/lora_qwen_course1.json  --resume