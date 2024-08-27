import os
import pdb
import torch
import numpy as np
import pandas as pd

def calculate_average_using_numpy(x, y, return_per_x=False):

    # 找出x值在6到18之间的索引
    valid_indices = (x >= 6) & (x <= 18)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    default = y_valid.min()
    # 对每个x值计算对应的y平均值
    unique_x = range(6, 19)
    average_y_per_x = dict()
    for xi in range(6, 19):
        if (x_valid == xi).sum() > 0:
            average_y_per_x.update({xi: np.mean(y_valid[x_valid == xi])})
        else:
            average_y_per_x.update({xi: default})

    # 计算所有平均值的平均
    overall_average = np.mean([average for average in average_y_per_x.values()])
    if return_per_x:
        return average_y_per_x, overall_average
    else:
        return overall_average

def intergrate_per_method(df):

    dynamic_frame = df['Inter_frame'].round()
    dynamic_segm  = df['Inter_segm'].round()
    dynamic_video = df['Video_level'].round()
    dynamic_total = dynamic_frame + dynamic_segm + dynamic_video

    scores = dict(
        motion_smoothness = df['MotionSmoothness'],
        visual_quality = df['DoverScore'],
        text2video_align= df['Alignment'],
        reality=df['Reality'],
    )
    video_names = df['video_path'].str[:4]
    for name in video_names.unique():
        idx = video_names == name
        print(f'*********name: {name}***********')
        for key in scores:
            result = calculate_average_using_numpy(dynamic_total[idx].to_numpy(), scores[key][idx].to_numpy())
            print(key, ':', result)
            
            
def intergrate_per_method_post_round(df):
    dynamic_frame = df['Inter_frame']
    dynamic_segm  = df['Inter_segm']
    dynamic_video = df['Video_level']
    dynamic_total = (dynamic_frame + dynamic_segm + dynamic_video)

    scores = dict(
        motion_smoothness = df['MotionSmoothness'],
        visual_quality = df['DoverScore'],
        text2video_align= df['Alignment'],
        reality=df['Reality'],
    )
    video_names = df['video_path'].str[:4]
    for name in video_names.unique():
        idx = video_names == name
        print(f'*********name: {name}***********')
        for key in scores:
            result = calculate_average_using_numpy(dynamic_total[idx].to_numpy(), scores[key][idx].to_numpy())
            print(key, ':', result)

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


def get_overall_dynamic_scores(dynamics_scores, regress_model_weight_path='/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/model_weight/linear_regress_model.pth'):
    regress_models = RegressionModel(regress_model_weight_path)
    inter_frame = np.stack([dynamics_scores['flow'], dynamics_scores['ssim'], dynamics_scores['phash']], axis=1)
    inter_segme = np.stack([dynamics_scores['dino_segm_dist'], dynamics_scores['viclip_segm_dist']], axis=1)
    video_level = np.stack([dynamics_scores['info_dino'], dynamics_scores['temporal_entropy']], axis=1)
    return dynamics_scores['video_name'], regress_models.predict(inter_frame, inter_segme, video_level)

def filter_dataframes_by_common_videos(dynamics_scores, df, video_name_col='video_name', video_basename_col='video_basename'):
    """
    根据两个DataFrame中的视频名称列的交集，过滤并返回仅包含交集部分的两个DataFrame。

    :param dynamics_scores: 包含视频评分的DataFrame。
    :param df: 包含其他视频相关数据的DataFrame。
    :param video_name_col: dynamics_scores中包含视频名称的列名。
    :param video_basename_col: df中包含视频名称的列名。
    :return: 两个过滤后的DataFrame（dynamics_scores_filtered, df_filtered）。
    """
    dynamics_scores.sort_values(by=video_name_col, inplace=True)
    df.sort_values(by=video_basename_col, inplace=True)
    
    dynamics_scores.drop_duplicates(subset=[video_name_col], inplace=True)

    # 计算交集
    dynamics_scores[video_name_col] = dynamics_scores[video_name_col].str.lower().str.split('.').str[0]
    df[video_basename_col] = df[video_basename_col].str.lower().str.split('.').str[0]
    df.drop_duplicates(subset=[video_basename_col], inplace=True)

    common_videos = pd.Index(dynamics_scores[video_name_col]).intersection(df[video_basename_col])
    # 过滤dynamics_scores和df，只保留交集中的条目
    filtered_dynamics_scores = dynamics_scores[dynamics_scores[video_name_col].isin(common_videos)]
    filtered_df = df[df[video_basename_col].isin(common_videos)]

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
            'Overall Mean': "{:.1f}".format(name_df['Overall Mean'].mean()),
            'Low Interval Mean': "{:.1f}".format(name_df['Low Interval Mean'].mean()),
            'Mid Interval Mean': "{:.1f}".format(name_df['Mid Interval Mean'].mean()),
            'High Interval Mean': "{:.1f}".format(name_df['High Interval Mean'].mean())
        }
        new_rows.append(average_row)

    # Append new rows to the original DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
    return results_df


def intergrate_per_method_auto_bin(df, dynamic_scores_path, keys=['MotionSmoothness', 'bg_consistency', 'Subject_Consistency', 'Reality'], name_col='video_path'):

    # dynamic_frame = df['Inter_frame']
    # dynamic_segm  = df['Inter_segm']
    # dynamic_video = df['Video_level']
    
    # dynamic_total = (dynamic_frame + dynamic_segm + dynamic_video) / 3
    dynamics_scores = pd.read_excel(dynamic_scores_path)
    # pdb.set_trace()
    dynamics_scores, df = filter_dataframes_by_common_videos(dynamics_scores, df, video_basename_col=name_col)
    
    video_names_dynamics_score, dynamic_total = get_overall_dynamic_scores(dynamics_scores)
    scores = dict()
    # pdb.set_trace()
    for key in keys:
        scores.update({key: df[key]})

    video_names_dynamics_score = video_names_dynamics_score.str[:4].str.lower()
    video_names = df[name_col].str[:4].str.lower()
    results_data = []
    for name in video_names.unique():
        idx = video_names == name
        idx_dynamic = video_names_dynamics_score == name
        for key in scores:
            # print(f'{key}, Simple Mean: {scores[key][idx].to_numpy().mean()}')
            result = calculate_mean_values(dynamic_total[idx_dynamic], scores[key][idx].to_numpy())
            result = np.array(result)
            assert len(result) % 3 == 0
            overall_mean = result.mean() * 100
            low_interval_mean = result[:len(result)//3].mean() * 100
            mid_interval_mean = result[len(result)//3 : -len(result)//3].mean() * 100
            high_interval_mean = result[-len(result)//3:].mean() * 100
            
            results_data.append({
                'Name': name,
                'Key': key,
                'Overall Mean': overall_mean,
                'Low Interval Mean': low_interval_mean,
                'Mid Interval Mean': mid_interval_mean,
                'High Interval Mean': high_interval_mean
            })
            # print(f'{key}, Blending positive Mean: {result[result != 0].mean()}')
        print(f'#------------------------#')
        results_df = pd.DataFrame(results_data)
        results_df = add_averages_to_results(results_df)
        results_df.to_excel('results-new/dynamics_quality_results.xlsx', index=False)
        print(f'results_df: {results_df}')
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

if __name__ == "__main__":
    # df = pd.read_excel(r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results\dyamic-with-quality-generated-videos-v1.xlsx")
    # keys=['motion_smoothness', 'background_consistency', 'subject_consistency', 'Reality']
    df =  pd.read_excel("/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/updated-dyamic-with-quality-annotated-videos.xlsx")
    dynamic_scores_path = '/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/DynamicResultsGeneratedVideos-mean-annotatedvidoes/merged_results.xlsx'
    # df = pd.read_excel("/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/quality_generated_eval_results_videos_merged_output.xlsx")
    # dynamic_scores_path = '/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/DynamicResultsGeneratedVideos-mean-genrated-vidoes/merged_results.xlsx'

    # keys = ['motion_smoothness', 'Reality', 'subject_consistency', 'bg_consistency']
    keys = ['MotionSmoothness', 'Reality', 'Subject_Consistency', 'bg_consistency']
    df['Reality'] -= df['Reality'].min()
    df['Reality'] /= df['Reality'].max()
    pdb.set_trace()
    intergrate_per_method_auto_bin(df, dynamic_scores_path, keys=keys, name_col='video_basename')
    