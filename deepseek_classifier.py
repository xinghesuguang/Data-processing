import pandas as pd
import numpy as np
import json
import requests
import os

from datetime import datetime

# TODO: 请在此处填写您的DeepSeek API Key

DEEPSEEK_API_KEY = "sk-b8d0a53e90a54ba39f171f1c9b64689b"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def load_data(file_path):
    """
    加载欠费用户历史数据
    """
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def aggregate_user_data(df):
    """
    按用户编号聚合数据，将多条记录合并为每个用户一条记录
    """
    if '用户编号' not in df.columns:
        print("警告：未找到'用户编号'列，无法进行用户聚合")
        return df
    
    print(f"聚合前数据：{len(df)}条记录")
    print(f"唯一用户数：{df['用户编号'].nunique()}个")
    
    # 数值列聚合方式：求和或平均
    numeric_cols = ['电费', '滞纳金', '欠费金额', '电量', '合同容量', '运行容量', '缴费次数']
    agg_dict = {}
    
    for col in numeric_cols:
        if col in df.columns:
            if col in ['缴费次数','欠费金额', '滞纳金', '电费', '电量']:
                agg_dict[col] = 'sum'  # 费用类和缴费次数求和
            elif col in ['合同容量', '运行容量']:
                agg_dict[col] = 'mean'  # 容量取平均

    # 按用户编号聚合
    user_df = df.groupby('用户编号').agg(agg_dict).reset_index()
    
    print(f"聚合后数据：{len(user_df)}条记录")
    return user_df

# 增强特征计算
def calculate_features(df):
    # 首先处理数据中的空值
    numeric_cols = ['电费', '滞纳金', '欠费金额', '电量', '合同容量', '运行容量', '缴费次数']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 基础特征：平均欠费金额
    df['avg_arrears'] = df['欠费金额'] if '欠费金额' in df.columns else 0
    
    # 基础特征：总延迟次数
    df['total_delays'] = (df['欠费金额'] > 0).astype(int) if '欠费金额' in df.columns else 0
    
    # 增强特征1：欠费金额占电费比例
    if '欠费金额' in df.columns and '电费' in df.columns:
        df['arrears_ratio'] = df['欠费金额'] / (df['电费'] + 1e-6)  # 避免除零
    else:
        df['arrears_ratio'] = 0
    
    # 增强特征2：滞纳金占欠费金额比例
    if '滞纳金' in df.columns and '欠费金额' in df.columns:
        df['penalty_ratio'] = df['滞纳金'] / (df['欠费金额'] + 1e-6)
    else:
        df['penalty_ratio'] = 0
    
    # 增强特征3：缴费频率（缴费次数/总月数，假设24个月）
    if '缴费次数' in df.columns:
        df['payment_frequency'] = df['缴费次数'] / 24
    else:
        df['payment_frequency'] = 0
    
    # 增强特征4：用电量稳定性（基于合同容量和运行容量）
    if '合同容量' in df.columns and '运行容量' in df.columns:
        df['capacity_utilization'] = df['运行容量'] / (df['合同容量'] + 1e-6)
    else:
        df['capacity_utilization'] = 1
    
    # 增强特征5：欠费严重程度（综合指标）
    df['arrears_severity'] = (df['avg_arrears'] * 0.4 + 
                             df['arrears_ratio'] * 100 * 0.3 + 
                             df['penalty_ratio'] * 100 * 0.3)
    
    # 处理计算后可能产生的无穷值和NaN值
    feature_cols = ['avg_arrears', 'total_delays', 'arrears_ratio', 
                   'penalty_ratio', 'payment_frequency', 'capacity_utilization', 
                   'arrears_severity']
    
    for col in feature_cols:
        if col in df.columns:
            # 替换无穷值为0
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            # 填充NaN值为0
            df[col] = df[col].fillna(0)
    
    return df

def prepare_data_for_api(user_data, batch_size=10):
    """
    准备发送给API的数据，分批处理
    """
    batches = []
    for i in range(0, len(user_data), batch_size):
        batch = user_data.iloc[i:i+batch_size]
        batches.append(batch)
    return batches

def create_classification_prompt(batch_data):
    """
    创建分类提示词
    """
    prompt = """
请根据用户欠费数据进行催费等级分类。

请根据以下数据特征自主判断每个用户的催费紧急程度，并分为三个等级：
- 轻度提醒：欠费情况较轻，可以温和提醒
- 中度催缴：欠费情况中等，需要积极催缴
- 重度追讨：欠费情况严重，需要强力追讨

请综合考虑欠费金额、欠费时长、滞纳金情况、缴费频率等因素，运用你的判断能力进行分类。

请严格按照以下JSON格式返回结果，不要添加任何其他文字：
{
  "classifications": [
    {"用户编号": "用户ID", "催费等级": "轻度提醒"},
    {"用户编号": "用户ID", "催费等级": "中度催缴"},
    {"用户编号": "用户ID", "催费等级": "重度追讨"}
  ]
}

用户数据：
"""
    
    for _, row in batch_data.iterrows():
        prompt += f"""
用户编号: {row['用户编号']}
欠费金额: {row['欠费金额']}
滞纳金: {row['滞纳金']}
电费: {row['电费']}
缴费次数: {row['缴费次数']}
合同容量: {row['合同容量']:.2f}
运行容量: {row['运行容量']:.2f}
欠费比例: {row['arrears_ratio']:.2f}
滞纳金比例: {row['penalty_ratio']:.2f}
缴费频率: {row['payment_frequency']:.2f}
容量利用率: {row['capacity_utilization']:.2f}
欠费严重程度: {row['arrears_severity']:.2f}
---
"""
    
    return prompt

def call_deepseek_api(prompt):
    """
    调用DeepSeek API进行分类
    """
    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        raise ValueError("请先在代码中设置您的DeepSeek API Key")
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 调试：打印原始响应内容
                print(f"API原始响应: {content[:200]}...")  # 只打印前200字符
                
                # 尝试解析JSON响应
                try:
                    # 提取JSON部分
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    else:
                        print(f"未找到JSON格式，完整响应: {content}")
                        raise ValueError("未找到有效的JSON响应")
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {e}")
                    print(f"尝试解析的内容: {content}")
                    raise ValueError("JSON解析失败")
                    
                break  # 成功则跳出重试循环
                
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2)  # 等待2秒后重试
            
    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {e}")
        return None
    except Exception as e:
        print(f"处理响应时出错: {e}")
        return None

def classify_users_with_deepseek(user_data):
    """
    使用DeepSeek API对用户进行分类
    """
    print("开始使用DeepSeek API进行用户分类...")
    
    # 分批处理数据
    batches = prepare_data_for_api(user_data, batch_size=10)
    all_classifications = []
    
    for i, batch in enumerate(batches):
        print(f"处理第 {i+1}/{len(batches)} 批数据...")
        
        # 创建提示词
        prompt = create_classification_prompt(batch)
        
        # 调用API
        result = call_deepseek_api(prompt)
        
        if result and 'classifications' in result:
            all_classifications.extend(result['classifications'])
            print(f"成功分类 {len(result['classifications'])} 个用户")
        else:
            print(f"第 {i+1} 批数据分类失败，使用默认分类")
            # 使用默认分类作为备选
            for _, row in batch.iterrows():
                if row['arrears_ratio'] > 0.7 or row['penalty_ratio'] > 0.3:
                    level = "重度追讨"
                elif row['arrears_ratio'] > 0.4 or row['penalty_ratio'] > 0.1:
                    level = "中度催缴"
                else:
                    level = "轻度提醒"
                
                all_classifications.append({
                    "用户编号": row['用户编号'],
                    "催费等级": level
                })
    
    return all_classifications

def save_results(user_data, classifications, output_file):
    """
    保存分类结果
    """
    # 将分类结果转换为DataFrame
    classifications_df = pd.DataFrame(classifications)
    
    # 确保用户编号列的数据类型一致
    if not classifications_df.empty:
        # 先转换为数值类型，处理字符串格式的数字（如"45.0"）
        classifications_df['用户编号'] = pd.to_numeric(classifications_df['用户编号'], errors='coerce')
        # 再转换为与原数据相同的类型
        classifications_df['用户编号'] = classifications_df['用户编号'].astype(user_data['用户编号'].dtype)
    
    # 合并用户数据和分类结果
    result_df = pd.merge(user_data, classifications_df, on='用户编号', how='left')
    
    # 填充缺失的分类
    result_df['催费等级'] = result_df['催费等级'].fillna('轻度提醒')
    
    # 保存结果
    result_df.to_excel(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    
    # 打印统计信息
    label_counts = result_df['催费等级'].value_counts()
    print("\n分类结果统计:")
    for label, count in label_counts.items():
        print(f"{label}: {count} 人 ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def main():
    """
    主函数
    """
    try:
        # 加载数据
        df = load_data('欠费用户历史数据.xlsx')
        if df is None:
            return
        
        # 聚合用户数据
        user_data = aggregate_user_data(df)
        
        # 计算增强特征
        user_data = calculate_features(user_data)
        
        # 使用DeepSeek API进行分类
        classifications = classify_users_with_deepseek(user_data)
        
        # 保存结果
        output_file = 'DeepSeek用户催费标签结果.xlsx'
        result_df = save_results(user_data, classifications, output_file)
        
        print(f"\nDeepSeek分类完成！共处理 {len(result_df)} 个用户")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()