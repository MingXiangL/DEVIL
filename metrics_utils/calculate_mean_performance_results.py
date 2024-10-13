import pandas as pd

def calculate_group_averages(file_path, alignment_column, metrics):
    """
    读取Excel文件，根据每行alignment_column列的值的前四个字符，对模型进行分组，
    计算每一组的metrics列的平均值。

    参数:
    - file_path: Excel文件的路径
    - alignment_column: 用于分组的列名
    - metrics: 需要计算平均值的列名列表

    返回:
    - DataFrame: 包含每个分组及其平均值的DataFrame
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 提取alignment_column列的前四个字符并作为新的分组列
    df['group'] = df[alignment_column].str[:4]

    # 计算每个分组的metrics列的平均值
    agg_dict = {metric: 'mean' for metric in metrics}
    grouped = df.groupby('group').agg(agg_dict).reset_index()

    return grouped

# 调用示例
file_path = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results-new\updated-hs-ms-s1-vc1-zs-dynamic_results.xlsx"
alignment_column = 'video_basename'
metrics = ['DoverScore', 'MotionSmoothness', 'Reality']

result = calculate_group_averages(file_path, alignment_column, metrics)
print(result)
