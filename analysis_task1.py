'''
使用二次微调后的模型进行金融文本情绪提取
'''
import pandas as pd
import os
import csv
import transformers
import torch
from transformers import  AutoTokenizer,AutoModelForCausalLM, GenerationConfig, pipeline
from peft import PeftModel
from tqdm import tqdm

model_id = "models/Base/Qwen-7B-Chat"
model_id = "models/Lora/Qwen_7B_Chat_lora_step1"
model_id = "models/Lora/Qwen_7B_Chat_lora_step2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16, 
    load_in_4bit = True,
    use_cache_quantization=True,
    use_cache_kernel=True, 
    trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False, 
    trust_remote_code=True
)

model.generation_config.temperature = 0.4
model.generation_config.max_new_tokens = 256
model.generation_config.top_p = 0.7


system_prompt = "你是一个中文文本分类器，判断以下金融文本的情绪类别，答案为积极、消极或者中性。"


# 遍历文件夹中的所有CSV文件
folder_path = "CSI300Data"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def process_file(file_path):
    df = pd.read_csv(file_path)
    
    if 'qwen_lora2_情绪提取' in df.columns:
        print(f"{file_path} already done")
        return
    
    # 解析日期列并过滤出2019年的数据
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df_2019 = df[df['Date'].dt.year == 2019].copy()

    data = df_2019['Title'].fillna('') + " " + df_2019['Content'].fillna('')
    
    responses = []
    for text in data:
        if text.strip() == "":
            responses.append("")
        else:
            response, history = model.chat(tokenizer, text, history=None, system=system_prompt)
            responses.append(response)
    
    df_2019['qwen_lora2_情绪提取'] = responses
    print(file_path)
    df_2019.to_csv(file_path, index=False, mode="w", quoting=csv.QUOTE_NONNUMERIC)
    # print(df_2019)

for csv_file in tqdm(csv_files, desc="Overall Progress"):
    file_path = os.path.join(folder_path, csv_file)
    process_file(file_path)



# use case:
#  CUDA_VISIBLE_DEVICES=2,3 python analysis_task1.py
    
    