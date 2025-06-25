import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta

# 数据加载
def load_data(file_path):
    try:
        # 尝试读取Sheet1
        df = pd.read_excel(file_path, sheet_name='欠费用户历史缴费数据')
    except ValueError:
        # 如果Sheet1不存在，读取第一个工作表
        print("未找到Sheet1，正在读取第一个工作表...")
        df = pd.read_excel(file_path, sheet_name=0)
    
    print(f"成功读取数据，共 {len(df)} 行")
    print(f"列名: {list(df.columns)}")
    return df

# 用户数据聚合
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
            if col in ['缴费次数']:
                agg_dict[col] = 'sum'  # 缴费次数求和
            elif col in ['合同容量', '运行容量', '欠费金额']:
                agg_dict[col] = 'mean'  # 容量取平均
            else:
                agg_dict[col] = 'sum'  # 其他费用类求和
    
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

# 数据标准化（使用增强特征）
def standardize_data(df):
    feature_cols = ['avg_arrears', 'total_delays', 'arrears_ratio', 
                   'penalty_ratio', 'payment_frequency', 'capacity_utilization', 
                   'arrears_severity']
    
    # 确保所有特征列都存在
    available_features = [col for col in feature_cols if col in df.columns]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    return scaled_data, available_features

# 最佳聚类数选择（改进版）
def select_best_k(scaled_data):
    best_score = -1
    best_k = 3  # 默认3个等级
    scores = []
    
    # 限制聚类数在3-10之间，避免过度细分
    for k in range(3, 10):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        scores.append((k, score))
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"聚类评分: {scores}")
    print(f"选择的最佳聚类数: {best_k}, 轮廓系数: {best_score:.3f}")
    return best_k

# 业务规则标签映射
def map_labels_with_business_rules(df, labels):
    # 基于K-means聚类结果进行三分类标签映射
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 计算每个聚类的平均风险水平（基于平均欠费金额）
    cluster_risk = {}
    for cluster in unique_labels:
        cluster_mask = labels == cluster
        if 'avg_arrears' in df.columns:
            cluster_risk[cluster] = df.loc[cluster_mask, 'avg_arrears'].mean()
        else:
            cluster_risk[cluster] = cluster  # 如果没有欠费数据，直接用聚类编号
    
    # 计算每个聚类的人数
    # 原逻辑（按风险排序）
    # sorted_clusters = sorted(cluster_risk.keys(), key=lambda x: cluster_risk[x])
    
    # 新逻辑（按人数降序排序）
    cluster_sizes = {cluster: sum(labels == cluster) for cluster in unique_labels}
    sorted_clusters = sorted(cluster_sizes.keys(), key=lambda x: -cluster_sizes[x])
    
    # 映射到三个催费等级
    if n_clusters == 3:
        # 直接映射三个聚类
        label_mapping = {
            sorted_clusters[0]: '轻度提醒',
            sorted_clusters[1]: '中度催缴', 
            sorted_clusters[2]: '重度追讨'
        }
    elif n_clusters == 2:
        # 两个聚类映射为轻度和重度
        label_mapping = {
            sorted_clusters[0]: '轻度提醒',
            sorted_clusters[1]: '重度追讨'
        }
    else:
        # 多于3个聚类时，分组映射到三个等级
        label_mapping = {}
        third = len(sorted_clusters) // 3
        
        for i, cluster in enumerate(sorted_clusters):
            if i < third:
                label_mapping[cluster] = '轻度提醒'
            elif i < 2 * third:
                label_mapping[cluster] = '中度催缴'
            else:
                label_mapping[cluster] = '重度追讨'
    
    # 应用聚类标签映射
    df['催费等级'] = [label_mapping[label] for label in labels]
    
    # 应用业务规则进行微调
    # 业务规则1：极高欠费金额用户确保为重度追讨
    if 'avg_arrears' in df.columns:
        high_arrears_threshold = df['avg_arrears'].quantile(0.9)
        df.loc[df['avg_arrears'] > high_arrears_threshold, '催费等级'] = '重度追讨'
    
    # 业务规则2：高滞纳金比例用户升级催费等级
    if 'penalty_ratio' in df.columns:
        high_penalty_threshold = 0.15  # 滞纳金超过欠费金额15%
        mask = df['penalty_ratio'] > high_penalty_threshold
        df.loc[mask & (df['催费等级'] == '轻度提醒'), '催费等级'] = '中度催缴'
        df.loc[mask & (df['催费等级'] == '中度催缴'), '催费等级'] = '重度追讨'
    
    # 业务规则3：低缴费频率用户升级催费等级
    if 'payment_frequency' in df.columns:
        low_frequency_threshold = 0.4  # 缴费频率低于40%
        mask = df['payment_frequency'] < low_frequency_threshold
        df.loc[mask & (df['催费等级'] == '轻度提醒'), '催费等级'] = '中度催缴'
    
    return df

# 主函数
def main():
    file_path = '欠费用户历史数据.xlsx'
    
    print("=== 电网企业欠费用户催费标签构建系统 ===")
    
    # 加载数据
    print("\n1. 加载数据...")
    df = load_data(file_path)
    
    # 用户数据聚合
    print("\n2. 用户数据聚合...")
    df = aggregate_user_data(df)
    
    # 计算增强特征
    print("\n3. 计算增强特征...")
    df = calculate_features(df)
    
    # 数据标准化
    print("\n4. 数据标准化...")
    scaled_data, feature_names = standardize_data(df)
    print(f"使用特征: {feature_names}")
    
    # 使用轮廓系数选择最佳聚类数
    print("\n5. 选择最佳聚类数...")
    best_k = select_best_k(scaled_data)
    
    # 聚类模型
    print(f"\n6. 执行K-means聚类 (k={best_k})...")
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    
    # 应用业务规则标签映射
    print("\n7. 应用业务规则进行标签映射...")
    df = map_labels_with_business_rules(df, labels)
    
    # 统计结果
    print("\n8. 催费等级分布:")
    level_counts = df['催费等级'].value_counts()
    for level, count in level_counts.items():
        percentage = count / len(df) * 100
        print(f"   {level}: {count}人 ({percentage:.1f}%)")
    
    # 结果输出
    print("\n9. 保存结果...")
    output_cols = ['用户编号', '催费等级', 'avg_arrears', 'arrears_ratio', 
                   'penalty_ratio', 'payment_frequency', 'arrears_severity']
    available_output_cols = [col for col in output_cols if col in df.columns]
    
    # 优先保存Excel格式
    excel_filename = 'k_means用户催费标签结果.xlsx'
    try:
        df[available_output_cols].to_excel(excel_filename, index=False)
        print(f"结果已保存到 '{excel_filename}'")
    except PermissionError:
        print(f"无法保存到 '{excel_filename}'，文件可能被占用。请关闭Excel文件后重试。")
        # 备用方案：保存为CSV格式
        csv_filename = '用户催费标签结果.csv'
        df[available_output_cols].to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV格式到 '{csv_filename}'")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        # 备用方案：保存为CSV格式
        csv_filename = '用户催费标签结果.csv'
        df[available_output_cols].to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV格式到 '{csv_filename}'")
    
    return df

if __name__ == "__main__":
    main()