"""
Data Utilities

This module contains useful functions for： 
    Preprocessing data for different model bases.

Author: huakyouin

Usage:
    import data_utils

    # Example usage:
    preprocessed_data = data_utils.preprocess_data_for_qwen(raw_data)
"""
import transformers
import torch
from transformers.trainer_pt_utils import LabelSmoother
from datasets import Dataset


## llama次元序列化（重构后尚未验证）
def preprocess_data_for_llama(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dataset:

    tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask, labels = [], [], []

    for i, source in enumerate(sources):
        instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{source['instruction'] + source['input']}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{source['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > max_len:  # 做一个截断
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    })


## qwen词元序列化
def preprocess_data_for_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 512,
) -> Dataset:

    IGNORE_TOKEN_ID = LabelSmoother.ignore_index
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        instruction_id = tokenizer(source['instruction']).input_ids
        text1_id = tokenizer(source["input"]).input_ids
        text2_id =  tokenizer(source["output"]).input_ids


        input_id = [im_start] + _system + instruction_id + [im_end] + nl_tokens
        input_id += [im_start] + _user + text1_id + [im_end] + nl_tokens
        input_id += [im_start] + _assistant + text2_id + [im_end] + nl_tokens

        target = [im_start] + [IGNORE_TOKEN_ID] * (len(_system)+len(instruction_id)) + [im_end] + nl_tokens
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(_user)+len(text1_id)) + [im_end] + nl_tokens
        target += [im_start] + [IGNORE_TOKEN_ID] * len(_assistant) + text2_id + [im_end] + nl_tokens
        
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    # for the use of *.ne
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return Dataset.from_dict(dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    ))