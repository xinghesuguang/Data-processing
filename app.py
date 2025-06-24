from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
from k_means import main as generate_kmeans_labels

# 导入XGBoost分类器
from xgboost_classifier import main as generate_xgboost_labels

# 导入DeepSeek分类器
from deepseek_classifier import main as generate_deepseek_labels

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_labels', methods=['POST'])
def generate_labels_api():
    try:
        # 获取分类方法参数，默认为xgboost
        method = request.json.get('method', 'xgboost') if request.is_json else 'xgboost'
        
        # 根据方法选择不同的结果文件名
        if method == 'xgboost':
            result_file = 'XGBoost用户催费标签结果.xlsx'
        elif method == 'deepseek':
            result_file = 'DeepSeek用户催费标签结果.xlsx'
        else:  # 使用kmeans
            result_file = 'k_means用户催费标签结果.xlsx'
        
        # 检查结果文件是否已存在
        if os.path.exists(result_file):
            # 文件已存在，直接读取
            df = pd.read_excel(result_file)
        else:
            # 文件不存在，运行标签生成程序
            if method == 'xgboost':
                generate_xgboost_labels()
            elif method == 'deepseek':
                generate_deepseek_labels()
            else:
                generate_kmeans_labels()
            
            # 读取生成的结果文件
            if os.path.exists(result_file):
                df = pd.read_excel(result_file)
            else:
                return jsonify({'success': False, 'error': f'标签生成失败，未找到结果文件 {result_file}'})
        
        # 统计各标签数量
        label_counts = df['催费等级'].value_counts().to_dict()
        
        # 转换为前端需要的格式，包含增强特征信息
        result = {
            'success': True,
            'method': method,  # 返回使用的方法
            'total_users': len(df),
            'label_distribution': label_counts,
            'users': df.to_dict('records'),
            'feature_summary': {
                'avg_arrears_mean': df['avg_arrears'].mean() if 'avg_arrears' in df.columns else 0,
                'arrears_ratio_mean': df['arrears_ratio'].mean() if 'arrears_ratio' in df.columns else 0,
                'penalty_ratio_mean': df['penalty_ratio'].mean() if 'penalty_ratio' in df.columns else 0,
                'payment_frequency_mean': df['payment_frequency'].mean() if 'payment_frequency' in df.columns else 0,
                'arrears_severity_mean': df['arrears_severity'].mean() if 'arrears_severity' in df.columns else 0
            }
        }
        
        # 如果是XGBoost方法，尝试获取特征重要性
        if method == 'xgboost':
            try:
                # 尝试重新运行XGBoost以获取特征重要性（不保存结果）
                import xgboost_classifier
                temp_df = xgboost_classifier.load_data('欠费用户历史数据.xlsx')
                temp_df = xgboost_classifier.aggregate_user_data(temp_df)
                temp_df = xgboost_classifier.calculate_features(temp_df)
                scaled_data, feature_names = xgboost_classifier.standardize_data(temp_df)
                temp_df = xgboost_classifier.generate_initial_labels(temp_df)
                model, feature_importance = xgboost_classifier.train_xgboost_model(scaled_data, temp_df['初始催费等级'])
                
                # 添加特征重要性到结果
                feature_imp_dict = {}
                for i, feature in enumerate(feature_names):
                    feature_imp_dict[feature] = float(feature_importance[i])
                result['feature_importance'] = feature_imp_dict
            except Exception as e:
                print(f"获取特征重要性时出错: {e}")
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_historical_data')
def get_historical_data():
    try:
        if os.path.exists('欠费用户历史数据.xlsx'):
            df = pd.read_excel('欠费用户历史数据.xlsx')
            
            # 获取分页参数
            page = request.args.get('page', 1, type=int)
            page_size = request.args.get('page_size', 500, type=int)
            
            # 基本统计信息
            total_records = len(df)
            total_pages = (total_records + page_size - 1) // page_size  # 向上取整
            columns = df.columns.tolist()
            
            # 数值列统计
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            stats = {}
            for col in numeric_cols:
                col_mean = df[col].mean()
                col_max = df[col].max()
                col_min = df[col].min()
                
                stats[col] = {
                    'mean': float(col_mean) if pd.notna(col_mean) else None,
                    'max': float(col_max) if pd.notna(col_max) else None,
                    'min': float(col_min) if pd.notna(col_min) else None
                }
            
            # 分页处理数据
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_data = df.iloc[start_idx:end_idx].fillna('')  # 将NaN替换为空字符串
            
            result = {
                'success': True,
                'total_records': total_records,
                'total_pages': total_pages,
                'current_page': page,
                'page_size': page_size,
                'columns': columns,
                'numeric_stats': stats,
                'data': page_data.to_dict('records')  # 返回当前页数据
            }
            
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': '未找到历史数据文件'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/compare_methods')
def compare_methods():
    try:
        # 检查三种方法的结果文件是否存在
        kmeans_file = 'k_means用户催费标签结果.xlsx'
        xgboost_file = 'XGBoost用户催费标签结果.xlsx'
        deepseek_file = 'DeepSeek用户催费标签结果.xlsx'
        
        results = {'success': True, 'methods': []}
        
        # 检查K-means结果
        if os.path.exists(kmeans_file):
            kmeans_df = pd.read_excel(kmeans_file)
            kmeans_stats = {
                'method': 'kmeans',
                'total_users': len(kmeans_df),
                'label_distribution': kmeans_df['催费等级'].value_counts().to_dict()
            }
            results['methods'].append(kmeans_stats)
        
        # 检查XGBoost结果
        if os.path.exists(xgboost_file):
            xgboost_df = pd.read_excel(xgboost_file)
            xgboost_stats = {
                'method': 'xgboost',
                'total_users': len(xgboost_df),
                'label_distribution': xgboost_df['催费等级'].value_counts().to_dict()
            }
            results['methods'].append(xgboost_stats)
        
        # 检查DeepSeek结果
        if os.path.exists(deepseek_file):
            deepseek_df = pd.read_excel(deepseek_file)
            deepseek_stats = {
                'method': 'deepseek',
                'total_users': len(deepseek_df),
                'label_distribution': deepseek_df['催费等级'].value_counts().to_dict()
            }
            results['methods'].append(deepseek_stats)
            
            # 如果有多种方法的结果，进行比较分析
            available_methods = []
            comparison_dfs = {}
            
            if os.path.exists(kmeans_file):
                kmeans_df = pd.read_excel(kmeans_file)
                comparison_dfs['kmeans'] = kmeans_df[['用户编号', '催费等级']].rename(columns={'催费等级': 'kmeans_标签'})
                available_methods.append('kmeans')
            
            if os.path.exists(xgboost_file):
                xgboost_df = pd.read_excel(xgboost_file)
                comparison_dfs['xgboost'] = xgboost_df[['用户编号', '催费等级']].rename(columns={'催费等级': 'xgboost_标签'})
                available_methods.append('xgboost')
            
            if os.path.exists(deepseek_file):
                deepseek_df = pd.read_excel(deepseek_file)
                comparison_dfs['deepseek'] = deepseek_df[['用户编号', '催费等级']].rename(columns={'催费等级': 'deepseek_标签'})
                available_methods.append('deepseek')
            
            # 如果有至少两种方法的结果，进行比较
            if len(available_methods) >= 2:
                # 合并所有可用的方法结果
                comparison_df = None
                for method in available_methods:
                    if comparison_df is None:
                        comparison_df = comparison_dfs[method]
                    else:
                        comparison_df = pd.merge(comparison_df, comparison_dfs[method], on='用户编号', how='inner')
                
                # 计算整体一致性（所有方法都一致的比例）
                if len(available_methods) >= 3:
                    # 三种方法都存在时，计算三方一致性
                    all_agree = True
                    for i in range(len(available_methods)):
                        for j in range(i+1, len(available_methods)):
                            col1 = f'{available_methods[i]}_标签'
                            col2 = f'{available_methods[j]}_标签'
                            all_agree = all_agree & (comparison_df[col1] == comparison_df[col2])
                    
                    agreement_rate = all_agree.mean() * 100
                    disagreement_count = int((~all_agree).sum())
                elif len(available_methods) == 2:
                    # 只有两种方法时，计算两方一致性
                    col1 = f'{available_methods[0]}_标签'
                    col2 = f'{available_methods[1]}_标签'
                    agreement_rate = (comparison_df[col1] == comparison_df[col2]).mean() * 100
                    disagreement_count = int((comparison_df[col1] != comparison_df[col2]).sum())
                else:
                    agreement_rate = 100.0
                    disagreement_count = 0
                
                # 找出不一致的样本
                disagreement_samples = []
                
                if len(available_methods) >= 2:
                    # 找出至少有一对方法不一致的样本
                    disagreement_mask = False
                    for i in range(len(available_methods)):
                        for j in range(i+1, len(available_methods)):
                            col1 = f'{available_methods[i]}_标签'
                            col2 = f'{available_methods[j]}_标签'
                            disagreement_mask = disagreement_mask | (comparison_df[col1] != comparison_df[col2])
                    
                    disagreements = comparison_df[disagreement_mask].copy()
                    
                    # 选择前15个不一致样本，确保包含所有需要的列
                    sample_columns = ['用户编号']
                    for method in available_methods:
                        sample_columns.append(f'{method}_标签')
                    
                    sample_disagreements = disagreements[sample_columns].head(15)
                    disagreement_samples = sample_disagreements.to_dict('records')
                
                # 添加比较结果
                results['comparison'] = {
                    'available_methods': available_methods,
                    'agreement_rate': agreement_rate,
                    'disagreement_count': disagreement_count,
                    'total_common_users': len(comparison_df),
                    'sample_disagreements': disagreement_samples
                }
        
        if not results['methods']:
            return jsonify({'success': False, 'error': '未找到任何方法的结果文件'})
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)