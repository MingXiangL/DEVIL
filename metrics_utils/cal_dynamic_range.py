import numpy as np
import pdb
import pandas as pd

def cal_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # 计算IQR
    IQR = Q3 - Q1

    # 定义阈值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 筛选不是离群点的数据
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # 计算动态范围
    dynamic_range_iqr = filtered_data.max() - filtered_data.min()
    return dynamic_range_iqr

def cal_p99_1(data):
    # 计算1%和99%的百分位数
    p1 = np.percentile(data, 1)
    p99 = np.percentile(data, 99)

    # 计算动态范围
    dynamic_range_percentile = p99 - p1
    return dynamic_range_percentile


def cal_std(data):
    mean = np.mean(data)
    std_dev = np.std(data)

    # 定义阈值
    lower_std = mean - 3 * std_dev
    upper_std = mean + 3 * std_dev

    # 筛选数据
    filtered_data_std = data[(data >= lower_std) & (data <= upper_std)]

    # 计算动态范围
    dynamic_range_std = filtered_data_std.max() - filtered_data_std.min()
    return dynamic_range_std

def cal_dynamic_range(inter_frame, inter_segm, video_level):
    final_score = (inter_frame + inter_segm + video_level) / 3
    return {'percent_99': cal_p99_1(final_score), 'IQR': cal_iqr(final_score), 'std': cal_std(final_score)}

def cal_dynamic_range_single(final_score):
    return {'percent_99': cal_p99_1(final_score), 'IQR': cal_iqr(final_score), 'std': cal_std(final_score)}


def batch_dynamic_range(filename, prefixes=['GEN2_', 'st2v_', 'fn_lavie_', 'opensora_', 'vc2_', 'pika_']):
    result_df = pd.read_excel(filename)
    dynamic_range_dict = dict()
    dynamic_range_dict_total = dict()
    for prefix in prefixes:
        idx = result_df['video_names'].str.contains(prefix, na=False)
        filtered_df = result_df[idx]
        dynamic_range_total = cal_dynamic_range(filtered_df['Inter_frame'], filtered_df['Inter_segm'], filtered_df['Video_level'])
        dynamic_range_dict_total.update({prefix: dynamic_range_total})
        for key in ['Inter_frame', 'Inter_segm', 'Video_level']:
            dynamic_range = cal_dynamic_range_single(filtered_df[key])
        
            dynamic_range_dict.update({prefix+key: dynamic_range})
    return dynamic_range_dict_total, dynamic_range_dict


if __name__ == "__main__":
    # filename = "dynamics_pred_total.xlsx"
    filename  = "/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/dynamics_each_frame-1.xlsx"
    result_df = pd.read_excel(filename)
    dynamic_range_dict = dict()
    dynamic_range_dict_total = dict()
    for prefix in ['GEN2_', 'st2v_', 'fn_lavie_', 'opensora_', 'vc2_', 'pika_']:
        idx = result_df['Video_names'].str.contains(prefix, na=False)
        filtered_df = result_df[idx]
        dynamic_range_total = cal_dynamic_range(filtered_df['Inter_frame'], filtered_df['Inter_segment'], filtered_df['Video_level'])
        dynamic_range_dict_total.update({prefix: dynamic_range_total})
        for key in ['Inter_frame', 'Inter_segment', 'Video_level']:
            dynamic_range = cal_dynamic_range_single(filtered_df[key])
        
            dynamic_range_dict.update({prefix+key: dynamic_range})

    print('*'*10, 'total', '*'*10)
    for key in dynamic_range_dict_total:
        print(key, ':', dynamic_range_dict_total[key])

    print('*'*10, 'Each Level', '*'*10)
    for key in dynamic_range_dict:
        print(key, ':', dynamic_range_dict[key])
