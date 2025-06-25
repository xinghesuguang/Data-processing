import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
            if col in ['缴费次数','欠费金额', '滞纳金', '电费', '电量']:
                agg_dict[col] = 'sum'  # 缴费次数求和
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

# 生成初始标签（使用简单规则）
def generate_initial_labels(df):
    """
    使用简单规则生成初始标签
    """
    # 使用欠费金额和滞纳金比例作为主要指标
    if 'avg_arrears' not in df.columns:
        df['avg_arrears'] = 0
    if 'penalty_ratio' not in df.columns:
        df['penalty_ratio'] = 0
    
    # 标准化欠费金额用于分类
    arrears_mean = df['avg_arrears'].mean()
    arrears_std = df['avg_arrears'].std() if df['avg_arrears'].std() > 0 else 1
    df['arrears_z'] = (df['avg_arrears'] - arrears_mean) / arrears_std
    
    # 基于欠费金额和滞纳金比例的简单规则分类
    labels = []
    for _, row in df.iterrows():
        if row['arrears_z'] > 1 or row['penalty_ratio'] > 0.15:
            labels.append('重度追讨')
        elif row['arrears_z'] > 0 or row['penalty_ratio'] > 0.05:
            labels.append('中度催缴')
        else:
            labels.append('轻度提醒')
    
    df['初始催费等级'] = labels
    return df

# 使用XGBoost进行三分类
def train_xgboost_model(X, y):
    # 将标签转换为数值
    label_mapping = {'轻度提醒': 0, '中度催缴': 1, '重度追讨': 2}
    y_numeric = np.array([label_mapping[label] for label in y])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )
    
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # 多分类
        num_class=3,                # 三个类别
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))
    
    # 特征重要性
    feature_importance = model.feature_importances_
    return model, feature_importance

# 业务规则标签映射
def map_labels_with_business_rules(df, labels):
    # 将数值标签映射回文本标签
    label_mapping = {0: '轻度提醒', 1: '中度催缴', 2: '重度追讨'}
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
    
    print("=== 电网企业欠费用户催费标签构建系统 (XGBoost版) ===\n")
    
    # 加载数据
    print("1. 加载数据...")
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
    
    # 生成初始标签（使用简单规则，不依赖K-means）
    print("\n5. 生成初始标签...")
    df = generate_initial_labels(df)
    
    # 训练XGBoost模型
    print("\n6. 训练XGBoost模型...")
    model, feature_importance = train_xgboost_model(scaled_data, df['初始催费等级'])
    
    # 使用XGBoost模型预测
    print("\n7. 使用XGBoost模型预测...")
    predictions = model.predict(scaled_data)
    
    # 应用业务规则标签映射
    print("\n8. 应用业务规则进行标签映射...")
    df = map_labels_with_business_rules(df, predictions)
    
    # 打印特征重要性
    print("\n9. 特征重要性:")
    for i, feature in enumerate(feature_names):
        print(f"   {feature}: {feature_importance[i]:.4f}")
    
    # 统计结果
    print("\n10. 催费等级分布:")
    level_counts = df['催费等级'].value_counts()
    for level, count in level_counts.items():
        percentage = count / len(df) * 100
        print(f"   {level}: {count}人 ({percentage:.1f}%)")
    
    # 结果输出
    print("\n11. 保存结果...")
    
    # 排除初始催费等级列和arrears_z列，与k_means.py保持一致
    # output_df = df.drop(columns=['初始催费等级', 'arrears_z'], errors='ignore')
    # 显示所有列
    output_df = df
    excel_filename = 'XGBoost用户催费标签结果.xlsx'
    try:
        output_df.to_excel(excel_filename, index=False)
        print(f"结果已保存到 '{excel_filename}'")
    except PermissionError:
        print(f"无法保存到 '{excel_filename}'，文件可能被占用。请关闭Excel文件后重试。")
        # 备用方案：保存为CSV格式
        csv_filename = 'XGBoost用户催费标签结果.csv'
        output_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV格式到 '{csv_filename}'")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        # 备用方案：保存为CSV格式
        csv_filename = 'XGBoost用户催费标签结果.csv'
        output_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV格式到 '{csv_filename}'")
    
    return df

if __name__ == "__main__":
    main()