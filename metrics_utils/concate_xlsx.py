import os
import glob
import pdb
import pandas as pd


def update_row_xlsx(row):
    if 'Video' in row:
        row['Video'] = row['Video'].replace('\\', '/').split('/')[-1]
    if 'video_path' in row:
        row['video_path'] = row['video_path'].replace('\\', '/').split('/')[-1]
    return row

def merge_rows_with_shared_basename(files, columns_to_keep=None):
    # 读取每个文件的'video_basename'列，并存入集合中
    basename_sets = []
    for file in files:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        print(f'file: {file}')
        try:
            if 'video_basename' in df:
                df['video_basename'] = df['video_basename'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
            elif 'video_name' in df:
                df['video_basename'] = df['video_name'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
            elif 'video_path' in df:
                df['video_basename'] = df['video_path'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
            elif 'Video' in df:
                df['video_basename'] = df.apply(update_row_xlsx, axis=1)['Video'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
            elif 'video_names' in df:
                df['video_basename'] = df['video_names'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
            else:
                raise NotImplementedError
        except BaseException as e:
            print(e)
            pdb.set_trace()
        set_basename = set(df['video_basename'])
        basename_sets.append(set_basename)
        
    
    # 找出所有文件中共有的'video_basename'
    shared_basenames = set.intersection(*basename_sets)
    # shared_basenames = set.union(*basename_sets)
    # 从各文件中读取数据，并只保留共有的'video_basename'
    dfs = []
    for file in files:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        if 'video_basename' in df:
            df['video_basename'] = df['video_basename'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
        elif 'video_name' in df:
            df['video_basename'] = df['video_name'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
        elif 'video_path' in df:
            df['video_basename'] = df['video_path'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
        elif 'video_names' in df:
            df['video_basename'] = df['video_names'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
        elif 'Video' in df:
            df['video_basename'] = df.apply(update_row_xlsx, axis=1)['Video'].apply(lambda x: x.lower().replace('.mp4', '') if isinstance(x, str) else x).dropna()
        else:
            raise NotImplementedError

        # df['video_basename'] = df['video_basename'].str.replace('.mp4', '', regex=False)
        filtered_df = df[df['video_basename'].isin(shared_basenames)]
        dfs.append(filtered_df)
    
    # 对每个共有的video_basename，合并不同文件中的行
    merged_rows = []
    for basename in shared_basenames:
        # 提取所有文件中该basename的第一行
        rows = [df[df['video_basename'] == basename].head(1) for df in dfs]
        # 合并这些行到一行
        
        merged_row = pd.concat(rows, axis=0).reset_index(drop=True)
        # 使用第一个有效值填充缺失值，并限制最终输出的列
        final_row = merged_row.apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA)
        merged_rows.append(final_row)

    # 创建最终的DataFrame
    final_df = pd.DataFrame(merged_rows)

    # 只选择需要保留的列（确保包含'video_basename'）
    if columns_to_keep is not None:
        final_columns = ['video_basename'] + [col for col in columns_to_keep if col in final_df.columns]
        final_df = final_df[final_columns].reset_index(drop=True)

    # 保存结果到新的Excel文件
    final_df.to_excel('dyamic-with-quality-generated-videos-v2.xlsx', index=False)

files = [
    r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\dyamic-with-quality-generated-videos-v1.xlsx",
    r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results-new\quality-generated-videos-concated.xlsx",
    # r"C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results\bg_consistency.xlsx"
]
# files = glob.glob(os.path.join(r'C:\Users\Liaomx\OneDrive - mails.ucas.edu.cn\科研\AIGC\VideoGeneration\EvaluationCodes\results-new', '*.xlsx'))
# columns_to_keep = ['Motion-Smoothness', 'ViClip-Score', 'warping_error', 'DoverScore']

# 调用函数
merge_rows_with_shared_basename(files)
