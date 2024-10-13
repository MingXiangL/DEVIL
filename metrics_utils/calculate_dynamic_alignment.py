import numpy as np
import pdb
import pandas as pd

# 读取Excel文件
def read_excel_and_score(filename):
    # 读取Excel文件
    df = pd.read_excel(filename)


    # 定义关键词到分数的映射
    keywords_to_scores = {
        '_static': 1,
        '_low': 2,
        '_medium': 3,
        '_high': 4,
        '_very_high': 5
    }

    # 存储每个视频的分数
    scores = []

    # 遍历每行的filename
    for filename in df['Video_names']:
        # 存储当前filename的最高分数
        max_score = 0

        # 检查每个部分是否包含关键词
        for keyword, score in keywords_to_scores.items():
            
            if keyword in filename:
                # 更新最高分数
                max_score = max(max_score, score)
        # 添加得分到列表
        scores.append(max_score)

    # 将分数列添加到DataFrame
    df['name_dynamic_scores'] = scores

    return df


def calc_winning_rate(prompt_dynamics, video_dynamics):

    video_dynamics = np.array(video_dynamics)
    prompt_dynamics = np.array(prompt_dynamics)

    x_degree_other = dict()
    y_degree_other = dict()
    for y in np.unique(prompt_dynamics):
        idx = prompt_dynamics == y
        x_degree_other.update({y: video_dynamics[~idx]})
        y_degree_other.update({y: prompt_dynamics[~idx]})

    winning_rates = []
    for x, y in zip(video_dynamics, prompt_dynamics):
        x_other = x_degree_other[y]
        y_other = y_degree_other[y]
        win_rate = ((x - x_other) * (y - y_other) > 0).mean()
        winning_rates.append(win_rate)
    return sum(winning_rates) / len(winning_rates)

def batch_dynamic_alignment(filename, prefixes=['GEN2_', 'st2v_', 'fn_lavie_', 'opensora_', 'vc2_', 'pika_']):
    dynamic_alignments = dict()
    result_df = read_excel_and_score(filename)
    for prefix in prefixes:
        idx = result_df['Video_names'].str.contains(prefix, na=False)
        filtered_df = result_df[idx]
        extimated_dynamic = filtered_df['Inter_frame'] + filtered_df['Inter_segm'] + filtered_df['Video_level']
        name_dynamics = filtered_df['name_dynamic_scores']
        dyal = calc_winning_rate(name_dynamics, extimated_dynamic)
        dynamic_alignments.update({prefix: dyal*100})
    return dynamic_alignments

def get_name_dynamic_scores(df, column_name='Video_names'):
    """
    从Excel文件中提取视频名称，根据名称中是否包含特定的动态级别关键词来分配对应的数值。
    
    参数:
    excel_path (str): Excel文件的路径。
    column_name (str): 包含视频名称的列名，默认为'Video_names'。
    
    返回:
    np.array: 包含动态级别数值的数组。
    """

    # 提取指定列
    Video_names = df[column_name]

    # 创建一个空的numpy数组，用于存放结果
    results = np.zeros(len(Video_names), dtype=int)

    # 定义关键词和对应的数值
    keywords = ['static', 'low', 'medium', 'very_high', 'high']
    values = [0, 1, 2, 4, 3,]

    # 逐行检查每个video_name
    for i, name in enumerate(Video_names):
        for keyword, value in zip(keywords, values):
            if keyword in name.lower():  # 转换为小写进行检查
                results[i] = value
                break  # 找到一个匹配后停止检查当前名称，避免重复计分

    return results



if __name__ == "__main__":
    filename = "/root/paddlejob/workspace/env_run/output/liaomingxiang/DevilBench/dynamics_each_frame-1.xlsx"
    dynamic_alignments = dict()
    result_df = read_excel_and_score(filename)
    for prefix in ['GEN2_', 'st2v_', 'fn_lavie_', 'opensora_', 'vc2_', 'pika_']:
        idx = result_df['Video_names'].str.contains(prefix, na=False)
        filtered_df = result_df[idx]
        extimated_dynamic = filtered_df['Inter_frame'] + filtered_df['Inter_segment'] + filtered_df['Video_level']
        if 'name_dynamic_scores' in filtered_df:
            name_dynamics = filtered_df['name_dynamic_scores']
        else:
            name_dynamics = get_name_dynamic_scores(filtered_df)
        dyal = calc_winning_rate(name_dynamics, extimated_dynamic)
        dynamic_alignments.update({prefix: dyal*100})
    print(f'dynamic_alignments: {dynamic_alignments}')