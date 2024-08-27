import torch
import pandas as pd
from cal_dynamic_range import batch_dynamic_range, cal_dynamic_range_single
from calculate_dynamic_alignment import batch_dynamic_alignment, calc_winning_rate
from calculate_integration import intergrate_per_method_post_round
import os
import pdb
import numpy as np
from matplotlib import pyplot as plt

def normalize_and_sigmoid(data, mean, std):
    """
    Normalize the data and apply the sigmoid function to scale it between 0 and 1.
    
    Parameters:
        data (numpy.ndarray): Input data array.
    
    Returns:
        numpy.ndarray: Scaled data using the sigmoid function after normalization.
    """
    # 计算均值和标准差
    
    # 标准化数据
    normalized_data = (data - mean) / std
    
    # 应用 Sigmoid 函数
    sigmoid_data = 1 / (1 + np.exp(-normalized_data))
    
    return sigmoid_data

def read_quality_scores(data):
    pdb.set_trace()
    pass

def get_dynamic_scores(data, model):
    frame_level = linear_regress(np.stack([data['flow'], data['ssim'], data['phash']], axis=1), model=model['inter_frame']['model'])
    segm_level = linear_regress(np.stack([data['dino_segm_dist'], data['viclip_segm_dist']], axis=1), model=model['inter_segment']['model'])
    video_level = linear_regress(np.stack([data['info_dino'], data['temporal_entropy']], axis=1), model=model['video_level']['model'])
    
    frame_level = (frame_level - model['inter_frame']['min']) / (model['inter_frame']['max'] - model['inter_frame']['min'])
    segm_level = (segm_level - model['inter_segment']['min']) / (model['inter_segment']['max'] - model['inter_frame']['min'])
    video_level = (video_level - model['video_level']['min']) / (model['video_level']['max'] - model['video_level']['min'])

    return frame_level, segm_level, video_level, (frame_level + segm_level + video_level)/3

def get_dynamic_scores_with_norm(data, model):
    frame_level = linear_regress(np.stack([data['flow'], data['ssim'], data['phash']], axis=1), model=model['inter_frame']['model'])
    segm_level = linear_regress(np.stack([data['dino_segm_dist'], data['viclip_segm_dist']], axis=1), model=model['inter_segment']['model'])
    video_level = linear_regress(np.stack([data['info_dino'], data['temporal_entropy']], axis=1), model=model['video_level']['model'])
    frame_level = (frame_level - model['inter_frame']['min']) / (model['inter_frame']['max'] - model['inter_frame']['min'])
    segm_level = (segm_level - model['inter_segment']['min']) / (model['inter_segment']['max'] - model['inter_frame']['min'])
    video_level = (video_level - model['video_level']['min']) / (model['video_level']['max'] - model['video_level']['min'])

    return frame_level, segm_level, video_level, (frame_level + segm_level + video_level)/3


def linear_regress(data, model):
    return model.predict(data)

def normalize_data(data, norm_factor):
    if 'flow' in data:
        data['flow'] = normalize_and_sigmoid(
            data['flow'], 
            mean=norm_factor['inter_frame']['flow_mean'], 
            std=norm_factor['inter_frame']['flow_std'])
    if 'temporal_entropy' in data:
        data['temporal_entropy'] = normalize_and_sigmoid(
            data['temporal_entropy'], 
            mean=norm_factor['video_level']['info_mean'], 
            std=norm_factor['video_level']['info_std'])

    return data

def post_process(data):
    data = data - 6 # start from 0
    return data

def draw_hist(total_dynamic, save_prefix=None):
    counts, bins, _ = plt.hist(total_dynamic, bins=100, color='#1f77b4',edgecolor='black', density=True, linewidth=0.1)
    colors = plt.cm.GnBu(1- (bins - min(bins)) / (max(bins) - min(bins)))
    # colors = plt.cm.Blues(1- (bins - min(bins)) / (max(bins) - min(bins)))

    # 清除之前的直方图
    plt.clf()

    # 重新绘制直方图，为每个bin根据其边界值着色
    plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors[:-1], edgecolor='black')

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(f'hist_{save_prefix}.png')
    plt.close('all')

def map_dynamic_to_scores(scores):
    pass

def deal_dynamic_level(levels):
    dynamic_levels = []
    for level in levels:
        if '_static_' in level.lower():
            dynamic_levels.append(1)
        elif '_low_' in level.lower():
            dynamic_levels.append(2)
        elif '_medium_' in level.lower():
            dynamic_levels.append(3)
        elif '_very_high_' in level.lower():
            dynamic_levels.append(5)
        elif '_high_' in level.lower():
            dynamic_levels.append(4)
        else:
            raise NotImplementedError
    return np.array(dynamic_levels)

def group_by_prefix(array, prefix_length=4):
    """
    根据数组中每个元素的前四个字符分组，并给出每组的索引。
    
    参数:
    array (np.ndarray): 包含字符串的numpy数组
    prefix_length (int): 用于分组的前缀长度，默认值为4
    
    返回:
    dict: 以前缀为键，索引列表为值的字典
    """
    groups = {}
    
    for idx, element in enumerate(array):
        prefix = element[:prefix_length]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(idx)
    
    return groups


if __name__ == "__main__":
    # 1. 把数据从各个拿出来
    # 2. 把模型读进来
    # 3. 把数据该归一化的归一化
    # 4. 把值存到另一个文件内
    # dynamic_xlsx = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationResults\zs-etc\hs-ms-s1-vc1-zs-dynamic_results.xlsx"
    # dynamic_xlsx = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results-new\quality-generated-videos.xlsx"
    dynamic_xlsx = '/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/DynamicResultsMaxSimVideos/merged_results.xlsx'
    # dynamic_xlsx = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results\dyamic-with-quality-annotated-videos.xlsx"
    # dynamic_xlsx = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results\predict_dynamic_merge_ann_1.xlsx"
    linear_model_path = "model_weight/linear_regress_model_normed.pth"
    

    dynamic_scores = pd.read_excel(dynamic_xlsx)
    # annotated_dynamic_level = deal_dynamic_level(dynamic_scores['video_basename'])
    # video_name_indicator = 'video_name' if 'video_name' in dynamic_scores else 'video_basename'
    video_name_indicator = dynamic_scores.columns[0]
    pdb.set_trace()
    annotated_dynamic_level = deal_dynamic_level(dynamic_scores[video_name_indicator])
    # quality_scores = pd.read_excel()
    if 'flow' in dynamic_scores:
        linear_model = torch.load(linear_model_path)
        frame_dynamic, segm_dynamic, video_dynamic, total_dynamic = get_dynamic_scores_with_norm(normalize_data(dynamic_scores, linear_model), model=linear_model)
    else:
        frame_dynamic, segm_dynamic, video_dynamic = dynamic_scores['Inter_frame'], dynamic_scores['Inter_segm'], \
            dynamic_scores['Video_level']
        total_dynamic = (frame_dynamic + segm_dynamic + video_dynamic)/3
    # dynamic_scores = post_process(dynamic_scores)
    # group_idxes = group_by_prefix(dynamic_scores['video_basename'])
    group_idxes = group_by_prefix(dynamic_scores[video_name_indicator])
    dynamic_range = dict()
    dynamic_align = dict()
    for key in group_idxes:
        idx = group_idxes[key]
        range_total = cal_dynamic_range_single(total_dynamic[idx])
        range_frame = cal_dynamic_range_single(frame_dynamic[idx])
        range_segm = cal_dynamic_range_single(segm_dynamic[idx])
        range_video = cal_dynamic_range_single(video_dynamic[idx])
        align = calc_winning_rate(annotated_dynamic_level[idx], total_dynamic[idx])
        dynamic_range[key] = {'total': range_total, 'frame': range_frame, 'segm': range_segm, 'video': range_video}
        dynamic_align[key] = align
    # draw_hist(dynamic_scores, 'msrvtt')
    for key in dynamic_range:
        print('*'*10, 'dynamic_range of ', key, '*'*10)
        for k in dynamic_range[key]:
            print(f'{k}: {dynamic_range[key][k]}')
        print(f'Dynamic Alignment: {dynamic_align[key]*100}')
        
    dynamic_scores['Inter_frame'] = frame_dynamic
    dynamic_scores['Inter_segm'] = segm_dynamic
    dynamic_scores['Video_level'] = video_dynamic
    dynamic_scores['Total_dynamic'] = total_dynamic
    dynamic_scores.to_excel(os.path.join('results-new', 'updated-' + os.path.basename(dynamic_xlsx)), index=False)

    # print(f'dynamic_align: {dynamic_align}')
