import json
import glob
import pandas as pd
from typing import List

def extract_and_save_jsons_to_excel(json_file_paths: List[str], excel_file_path: str):
    """
    从多个JSON文件中提取特定内容并保存到一个Excel文件。

    :param json_file_paths: JSON文件路径的列表
    :param excel_file_path: 生成的Excel文件的路径
    """
    all_data = []

    # 遍历所有的JSON文件
    for json_file_path in json_file_paths:
        # 读取JSON文件
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # 遍历JSON文件中的每个key
        for key, value in data.items():
            # 检查是否是需要的结构
            if isinstance(value, list) and len(value) > 1 and isinstance(value[1], list):
                subject_consistency = value[1]
                for item in subject_consistency:
                    # 可以在这里添加其他信息以区分不同的key
                    item['key'] = key
                    item['source_file'] = json_file_path  # 添加来源文件信息
                    all_data.append(item)

    # 创建一个DataFrame
    df = pd.DataFrame(all_data)

    # 保存到Excel文件
    df.to_excel(excel_file_path, index=False)

# 使用示例
json_file_paths = glob.glob(r'C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationResults\generated_videos_zs\zs_reality_Results\*.json')
# json_file_paths = [r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationResults\generated_videos_zs\zs-motion_smoothness.json"]
# json_file_paths = [r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationResults\AnnotateVideos\annotated_videos_subject_background_consistency.json"]
excel_file_path = r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results-new\realism.xlsx"

extract_and_save_jsons_to_excel(json_file_paths, excel_file_path)