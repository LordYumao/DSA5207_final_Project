import pandas as pd
import os
import numpy as np
from pathlib import Path

def convert_txt_to_single_label_tsv(input_file, output_file, is_test_set=False):
    """
    将txt格式的多标签情感数据集转换为tsv格式的单标签数据集
    
    Args:
        input_file (str): 输入文件路径（txt格式）
        output_file (str): 输出文件路径（tsv格式）
        is_test_set (bool): 是否为测试集（无标签）
    """
    print(f"处理文件: {input_file}")
    
    # 读取txt文件为DataFrame
    df = pd.read_csv(input_file, sep='\t')
    
    if is_test_set:
        # 测试集：无标签，直接保存为要求的格式
        result_df = df[['ID', 'Tweet']]
        result_df.columns = ['id', 'text']  # 重命名列
        result_df.to_csv(output_file, sep='\t', index=False)
        print(f"测试集数据已保存至: {output_file}")
        return
    
    # 非测试集：处理标签
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 
                      'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    
    # 确保所有标签列都是数值类型
    for label in emotion_labels:
        if df[label].dtype == object:  # 如果是字符串类型
            try:
                df[label] = df[label].replace('NONE', '0').astype(int)
            except:
                print(f"警告: 无法将列 {label} 转换为数值类型")
    
    # 计算每行有多少个情感标签（值为1的标签数量）
    df['label_count'] = df[emotion_labels].sum(axis=1)
    
    # 筛选只有一个情感标签的行
    single_label_df = df[df['label_count'] == 1].copy()
    
    print(f"总样本数: {len(df)}")
    print(f"单标签样本数: {len(single_label_df)}")
    
    # 如果没有单标签样本，创建一个空的DataFrame并保存
    if len(single_label_df) == 0:
        result_df = pd.DataFrame(columns=['id', 'text', 'label'])
        result_df.to_csv(output_file, sep='\t', index=False)
        print(f"没有找到单标签样本，已创建空文件: {output_file}")
        return
    
    # 为每一行确定其唯一的情感标签
    single_label_df['emotion'] = ''
    for idx, row in single_label_df.iterrows():
        for emotion in emotion_labels:
            if row[emotion] == 1:
                single_label_df.at[idx, 'emotion'] = emotion
                break
    
    # 选择需要的列并保存为tsv
    result_df = single_label_df[['ID', 'Tweet', 'emotion']]
    result_df.columns = ['id', 'text', 'label']  # 重命名列以便使用
    
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"单标签数据已保存至: {output_file}")

def process_dataset(input_dir, output_dir):
    """
    处理整个数据集（训练集、开发集、测试集）
    
    Args:
        input_dir (str): 输入目录（txt文件）
        output_dir (str): 输出目录（tsv文件）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集和开发集（有标签）
    for file_name in ['2018-E-c-En-train.txt', '2018-E-c-En-dev.txt']:
        input_file = os.path.join(input_dir, file_name)
        if os.path.exists(input_file):
            base_name = os.path.basename(input_file).replace('.txt', '')
            output_file = os.path.join(output_dir, f"{base_name}_single_label.tsv")
            convert_txt_to_single_label_tsv(input_file, output_file, is_test_set=False)
    
    # 处理测试集（无标签）
    test_file = os.path.join(input_dir, '2018-E-c-En-test.txt')
    if os.path.exists(test_file):
        output_file = os.path.join(output_dir, '2018-E-c-En-test.tsv')
        convert_txt_to_single_label_tsv(test_file, output_file, is_test_set=True)

def main():
    # 配置参数
    input_dir = "data/semeval2018_ec"  # 原始txt数据集目录
    output_dir = "data/semeval2018_ec_single"  # 输出tsv目录
    
    # 处理数据集
    process_dataset(input_dir, output_dir)
    
    print("数据处理完成!")

if __name__ == "__main__":
    main()