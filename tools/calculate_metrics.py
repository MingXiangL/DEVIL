import os
import pdb
import torch
import numpy as np
import pandas as pd
import argparse
from prettytable import PrettyTable

from tools.evaluate_dynamic_ddp_metrics import cal_dynamics_metircs


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference with a pretrained DINO model on DiDemo dataset.")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save extracted features.')
    parser.add_argument('--regress_model_weight_path', type=str, default='model_weights/linear_regress_model.pth')
    parser.add_argument('--dynamic_score_save_name', type=str, default='dynamics_merged_results.xlsx')
    parser.add_argument('--quality_save_name', type=str, default='merged_results.xlsx')
    parser.add_argument('--print_detail_quality_results', action='store_true', default=False, help='Enable detailed quality results printing.')
    return parser.parse_args()


class RegressionModel:
    def __init__(self, model_path):
        # Load models from a .pth file
        self.models = torch.load(model_path)
    def predict(self, input_inter_frame, input_inter_segment, input_video_level):
        # Predict using the loaded models and clamp the values
        prediction_inter_frame = np.clip(self.models['inter_frame'].predict(input_inter_frame), 0, 1)
        prediction_inter_segment = np.clip(self.models['inter_segment'].predict(input_inter_segment), 0, 1)
        prediction_video_level = np.clip(self.models['video_level'].predict(input_video_level), 0, 1)
        # Calculate the mean of predictions for each entry
        return np.mean([prediction_inter_frame, prediction_inter_segment, prediction_video_level], axis=0)


def get_overall_dynamic_scores(dynamics_scores, regress_model_weight_path):
    regress_models = RegressionModel(regress_model_weight_path)
    inter_frame = np.stack([dynamics_scores['flow'], dynamics_scores['ssim'], dynamics_scores['phash']], axis=1)
    inter_segme = np.stack([dynamics_scores['dino_segm_dist'], dynamics_scores['viclip_segm_dist']], axis=1)
    video_level = np.stack([dynamics_scores['info_dino'], dynamics_scores['temporal_entropy']], axis=1)
    return dynamics_scores['video_name'], regress_models.predict(inter_frame, inter_segme, video_level)


def filter_dataframes_by_common_videos(df_dynamics, df_quality, video_name_col='video_name', video_basename_col='video_name'):
    """
    根据两个DataFrame中的视频名称列的交集，过滤并返回仅包含交集部分的两个DataFrame。

    :param df_dynamics: 包含视频评分的DataFrame。
    :param df_quality: 包含其他视频相关数据的DataFrame。
    :param video_name_col: dynamics_scores中包含视频名称的列名。
    :param video_basename_col: df中包含视频名称的列名。
    :return: 两个过滤后的DataFrame（dynamics_scores_filtered, df_filtered）。
    """
    df_dynamics.sort_values(by=video_name_col, inplace=True)
    df_quality.sort_values(by=video_basename_col, inplace=True)
    
    df_dynamics.drop_duplicates(subset=[video_name_col], inplace=True)

    # 计算交集
    df_dynamics[video_name_col] = df_dynamics[video_name_col].str.lower().str.split('.').str[0]
    df_quality[video_basename_col] = df_quality[video_basename_col].str.lower().str.split('.').str[0]
    df_quality.drop_duplicates(subset=[video_basename_col], inplace=True)

    common_videos = pd.Index(df_dynamics[video_name_col]).intersection(df_quality[video_basename_col])
    # 过滤dynamics_scores和df，只保留交集中的条目
    filtered_dynamics_scores = df_dynamics[df_dynamics[video_name_col].isin(common_videos)]
    filtered_df = df_quality[df_quality[video_basename_col].isin(common_videos)]

    return filtered_dynamics_scores, filtered_df

def add_averages_to_results(results_df):
    unique_names = results_df['Name'].unique()
    new_rows = []  # List to store new rows for averages

    for name in unique_names:
        name_df = results_df[results_df['Name'] == name]
        average_row = {
            'Name': name + '_avg',
            'Key': 'Average',
            # 'Overall Mean': name_df['Overall Mean'].mean(),
            # 'Low Interval Mean': name_df['Low Interval Mean'].mean(),
            # 'Mid Interval Mean': name_df['Mid Interval Mean'].mean(),
            # 'High Interval Mean': name_df['High Interval Mean'].mean()
            'Overall Mean': name_df['Overall Mean'].mean(),
            'Low Interval Mean': name_df['Low Interval Mean'].mean(),
            'Mid Interval Mean': name_df['Mid Interval Mean'].mean(),
            'High Interval Mean': name_df['High Interval Mean'].mean()
        }
        new_rows.append(average_row)

    # Append new rows to the original DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
    return results_df


def intergrate_per_method_auto_bin(args, df_quality, df_dynamics, keys=['motion_smoothness', 'naturalness', 'subject_consistency', 'background_consistency'], name_col='video_name'):
# def intergrate_per_method_auto_bin(args, df_quality, df_dynamics, keys=['subject_consistency', 'background_consistency'], name_col='video_name'):

    df_dynamics, df_quality = filter_dataframes_by_common_videos(df_dynamics, df_quality, video_basename_col=name_col)
    
    video_names_dynamics_score, dynamic_total = get_overall_dynamic_scores(df_dynamics, args.regress_model_weight_path)
    scores = dict()

    for key in keys:
        scores.update({key: df_quality[key]})

    video_names_dynamics_score = video_names_dynamics_score.str[:4].str.lower()
    video_names = df_quality[name_col].str[:4].str.lower()
    results_data = []
    # for name in video_names.unique():
        # idx = video_names == name
    # idx_dynamic = video_names_dynamics_score == name
    for key in scores:
        # print(f'{key}, Simple Mean: {scores[key][idx].to_numpy().mean()}')
        result = calculate_mean_values(dynamic_total, scores[key].to_numpy())
        result = np.array(result)
        
        assert len(result) % 3 == 0
        overall_mean = result.mean()
        org_mean = scores[key].to_numpy().mean()
        mse_dynamics = np.sqrt(np.mean((result - overall_mean) ** 2))
        mse_oring    = np.sqrt(np.mean((result - org_mean) ** 2))
        # print('-' * 10, f'Metric: {key}', '-' * 10)
        # print(f'mse_dynamics: {mse_dynamics}')
        # print(f'mse_oring: {mse_oring}')
        # print('-' * 20)
        low_interval_mean = result[:len(result)//3].mean()
        mid_interval_mean = result[len(result)//3 : -len(result)//3].mean()
        high_interval_mean = result[-len(result)//3:].mean()
        
        results_data.append({
            'Name': 'Model',
            'Key': key,
            'Overall Mean': overall_mean,
            'Low Interval Mean': low_interval_mean,
            'Mid Interval Mean': mid_interval_mean,
            'High Interval Mean': high_interval_mean
        })
            # print(f'{key}, Blending positive Mean: {result[result != 0].mean()}')
        results_df = pd.DataFrame(results_data)
        results_df = add_averages_to_results(results_df)
        results_df.to_excel(f'{args.save_dir}/dynamics_quality_results.xlsx', index=False)
    return results_df
            # print(key, ':', result)


def calculate_mean_values(x, y, num_bins=12, x_min=0, x_max=1, default_value=0):
    # 检查x和y的长度是否匹配                
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")
    
    # 生成等间距的区间边界
    bins = np.linspace(x_min, x_max, num_bins + 1)
    
    # 使用digitize找到每个x值所属的区间
    bin_indices = np.digitize(x, bins)
    
    # 计算每个区间的y值均值
    y_means = []
    for i in range(1, num_bins + 1):
        # 选择当前区间内的y值
        indices = bin_indices == i
        y_values = y[indices]
        
        # 计算均值，如果当前区间没有数据，则返回NaN
        if y_values.size > 0:
            mean_value = np.mean(y_values)
        else:
            mean_value = default_value
            
        y_means.append(mean_value)
    
    return y_means

def keep_common_rows(df1, df2, column='video_name'):
    common_rows_df1 = df1[df1[column].isin(df2[column])].reset_index(drop=True)
    common_rows_df2 = df2[df2[column].isin(df1[column])].reset_index(drop=True)
    return common_rows_df1, common_rows_df2

def calculate_dynamics_based_quality(args):
    quality_save_path = os.path.join(args.save_dir, args.quality_save_name)
    dynamic_scores_path =  os.path.join(args.save_dir, args.dynamic_score_save_name)
    df_quality =  pd.read_excel(quality_save_path)
    df_dynamics = pd.read_excel(dynamic_scores_path)
    df_quality_unique_sorted = df_quality.drop_duplicates(subset='video_name').sort_values('video_name').reset_index(drop=True)
    df_dynamics_unique_sorted = df_dynamics.drop_duplicates(subset='video_name').sort_values('video_name').reset_index(drop=True)
    assert df_quality_unique_sorted['video_name'].equals(df_dynamics_unique_sorted['video_name'])
    dynamics_based_quality = intergrate_per_method_auto_bin(args, df_quality_unique_sorted, df_dynamics_unique_sorted)
    return dynamics_based_quality

def format_value(value):
    return "{:.2f}".format(float(value)*100)


def print_results(metrics):
    table = PrettyTable()
    table.field_names = ['Metric', 'Value (%)']

    for key, value in metrics.items():
        # 在添加行之前格式化数值
        formatted_value = format_value(value)
        table.add_row([key, formatted_value])
        
    print(table)


def print_detail_quality_results(quality_metrics):
    table = PrettyTable()
    table.field_names = ['Metric', 'Overall (%)', 'Low (%)', 'Mid (%)', 'High (%)']
    for _, row in quality_metrics.iterrows():
        metric = row['Key']
        overall = format_value(row['Overall Mean'])
        low = format_value(row['Low Interval Mean'])
        mid = format_value(row['Mid Interval Mean'])
        high= format_value(row['High Interval Mean'])
        table.add_row([metric, overall, low, mid, high])
    print(table)
    
if __name__ == '__main__':
    args = parse_args()
    dynamics_scores, dynamics_range, dynamics_controllability = cal_dynamics_metircs(args)
    dynamics_based_qualities = calculate_dynamics_based_quality(args)
    print_results({
        'Dynamics Range': dynamics_range, 
        'Dynamics Controllability': dynamics_controllability, 
        'Dynamics-based Quality Overall': dynamics_based_qualities[dynamics_based_qualities['Key']=='Average']['Overall Mean'].item(),
        'Dynamics-based Quality Low': dynamics_based_qualities[dynamics_based_qualities['Key']=='Average']['Low Interval Mean'].item(),
        'Dynamics-based Quality Mid': dynamics_based_qualities[dynamics_based_qualities['Key']=='Average']['Mid Interval Mean'].item(),
        'Dynamics-based Quality High': dynamics_based_qualities[dynamics_based_qualities['Key']=='Average']['High Interval Mean'].item(),
        })

    if args.print_detail_quality_results:
        print_detail_quality_results(dynamics_based_qualities)

