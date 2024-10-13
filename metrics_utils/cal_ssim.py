import os
import pdb
import cv2
import math
import shutil
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import subprocess
from tqdm import tqdm
import numpy as np
import imagehash
from PIL import Image
from pytorch_msssim import ms_ssim
from pytorch_msssim import ssim as torch_ssim
import torch

def count_png_files(directory):
    """
    递归统计指定文件夹及其子文件夹中PNG文件的数量。
    :param directory: 要统计的文件夹路径
    :return: PNG文件的数量
    """
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.png'):
            count += 1
    return count

def convert_to_h264(input_video, output_video):
    """
    将视频文件重新编码为H.264格式。
    :param input_video: 输入视频的文件路径
    :param output_video: 输出视频的文件路径
    """
    command = [
        'ffmpeg',
        '-i', input_video,               # 输入文件
        '-c:v', 'libx264',               # 指定视频编解码器为libx264（H.264）
        '-preset', 'fast',               # 压缩预设，可调整（ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow）
        output_video,                     # 输出文件
        '-y',
        '-loglevel', 'error'
    ]
    try:
        subprocess.run(command, check=True)
        # print("视频已成功重新编码为H.264格式。")
    except subprocess.CalledProcessError as e:
        print(f"发生错误：{e}")
        exit(0)


def read_frames(video_path):
    """
    逐帧读取视频。
    
    参数:
        video_path (str): 视频文件的路径。
    
    返回:
        generator: 返回帧的生成器。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()

def calculate_frame_ssim(frame1, frame2):
    """
    计算两帧之间的SSIM。
    
    参数:
        frame1, frame2 (numpy.ndarray): 需要比较的两帧图像。
    
    返回:
        float: 两帧之间的SSIM值。
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=gray2.max() - gray2.min())

def extract_keyframes(input_video, output_folder):
    """
    使用FFmpeg从视频中提取关键帧并保存为图像。
    :param input_video: 输入视频的文件路径
    :param output_folder: 输出图像的文件夹
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', input_video,            # 输入文件
        '-vf', 'select=eq(pict_type\\,I)',  # 选择I帧（关键帧）
        '-vsync', 'vfr',              # 使用变帧率
        '-y', '-loglevel', 'error',
        f'{output_folder}/keyframe_%03d.png'  # 输出文件格式和路径
    ]
    try:
        subprocess.run(command, check=True)
        print("关键帧提取完成。")
    except subprocess.CalledProcessError as e:
        print(f"发生错误：{e}")
        exit(0)

    count = count_png_files(output_folder)

    
    return count

def extract_keyframes_and_last_frame(input_video, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    # 提取关键帧
    command_keyframes = [
        'ffmpeg',
        '-i', input_video,
        '-vf', 'select=eq(pict_type\\,I)',
        '-vsync', 'vfr', '-y',
        '-loglevel', 'error',
        f'{output_folder}/keyframe_%03d.png'
    ]
    subprocess.run(command_keyframes)

    # 确定最后一个关键帧的编号
    key_frames = sorted([f for f in os.listdir(output_folder) if f.startswith('keyframe_') and f.endswith('.png')])
    last_index = int(key_frames[-1].split('_')[1].split('.')[0]) if key_frames else 0
    # 获取总帧数
    if len(key_frames) == 1:
        command_count_frames = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-loglevel', 'error',
            '-of', 'csv=p=0',
            input_video
        ]
        total_frames = subprocess.run(command_count_frames, capture_output=True, text=True)
        total_frames = int(total_frames.stdout.strip())
    
        command_last_frame = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f"select='eq(n\\,{total_frames-1})'",
            '-vframes', '1',
            '-loglevel', 'error',
            f'{output_folder}/keyframe_{last_index+1:03d}.png'
        ]
        subprocess.run(command_last_frame)


def get_image_files(directory):
    """ 获取指定目录下所有的PNG文件路径 """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

def load_images(image_folder):
    """ 加载图像文件并转换为灰度图 """
    file_paths = get_image_files(image_folder)

    images = []
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def calculate_ssim(frames):
    ssim_values = []
    for i in range(len(frames) - 1):
        ssim_value = ssim(frames[i], frames[i+1], data_range=frames[i+1].max() - frames[i+1].min())
        ssim_values.append(ssim_value)
    return sum(ssim_values) / len(ssim_values)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def stat_inter_keyframe_ssim(video_path, key_frame_folder):
    output_path = os.path.join('outputs', os.path.basename(video_path))
    convert_to_h264(video_path, output_path)
    key_frame_numbers = extract_keyframes_and_last_frame(output_path, output_folder=key_frame_folder)
    images = load_images(key_frame_folder)
    ssim_scores = calculate_ssim(images)
    return ssim_scores

def calculate_phash(image):
    return imagehash.phash(Image.fromarray(image))

def calculate_whash(image):
    return imagehash.whash(Image.fromarray(image))

def stat_ssim_with_path(video_path):
    """
    处理视频，打印每一帧与下一帧之间的SSIM值。
    
    参数:
        video_path (str): 视频文件的路径。
    """
    frame_generator = read_frames(video_path)
    prev_frame = next(frame_generator, None)
    if prev_frame is None:
        return
    prev_phash = calculate_phash(prev_frame)
    prev_whash = calculate_whash(prev_frame)

    frame_count = 0
    ssims = []
    psnrs = []
    phashs= []
    whashs= []
    for current_frame in frame_generator:
        current_ssim = calculate_frame_ssim(prev_frame, current_frame)
        current_psnr = calculate_psnr(prev_frame, current_frame)
        current_phash = calculate_phash(current_frame)
        current_whash = calculate_whash(current_frame)

        phash_diff = current_phash - prev_phash if prev_phash is not None else 0
        whash_diff = current_whash - prev_whash if prev_whash is not None else 0

        prev_phash = current_phash
        prev_whash = current_whash
        frame_count += 1
        prev_frame = current_frame
        ssims.append(current_ssim)
        psnrs.append(current_psnr)
        phashs.append(phash_diff)
        whashs.append(whash_diff)
    return sum(ssims) / len(ssims), sum(psnrs) / len(psnrs), sum(phashs) / len(phashs), sum(whashs) / len(whashs)


def stat_ssim(video_data, topk=0.1):
    """
    处理视频，打印每一帧与下一帧之间的SSIM值。
    
    参数:
        video_path (str): 视频文件的路径。
    """
    prev_frame = video_data[0]
    prev_phash = calculate_phash(prev_frame)
    prev_whash = calculate_whash(prev_frame)

    frame_count = 0
    ssims = []
    phashs= []
    
    for current_frame in video_data[1:]:
        current_ssim = calculate_frame_ssim(prev_frame, current_frame)
        current_phash = calculate_phash(current_frame)
        current_whash = calculate_whash(current_frame)

        phash_diff = current_phash - prev_phash if prev_phash is not None else 0
        whash_diff = current_whash - prev_whash if prev_whash is not None else 0

        prev_phash = current_phash
        prev_whash = current_whash
        frame_count += 1
        prev_frame = current_frame
        ssims.append(current_ssim)
        phashs.append(phash_diff)
    return sum(ssims) / len(ssims), sum(phashs) / len(phashs)


def stat_phash(video_data, average=True):
    """
    处理视频，打印每一帧与下一帧之间的SSIM值。
    
    参数:
        video_path (str): 视频文件的路径。
    """
    prev_frame = video_data[0]
    prev_phash = calculate_phash(prev_frame)

    phashs= []
    
    for current_frame in video_data[1:]:
        current_phash = calculate_phash(current_frame)
        phash_diff = current_phash - prev_phash if prev_phash is not None else 0
        prev_phash = current_phash
        prev_frame = current_frame
        phashs.append(phash_diff)
    if average:
        return sum(phashs) / len(phashs)
    else:
        return phashs

# def cal_ssim_dist(video_data, video_names):
#     ssims = []
#     msssims=[]
#     psnrs = []
#     phashs= []
#     results = dict()
#     for i, video in enumerate(video_data):
#         # print(f'calculating ssims for {video_names[i]}')
#         # phash = stat_phash(video.permute(0,2,3,1).to(torch.uint8).cpu().numpy())
#         phash = 0
#         print(f'video.shape: {video.shape}')
#         ssim = torch_ssim(video[:-1], video[1:], data_range=255, size_average=True)
#         # msssim = ms_ssim(video[:-1], video[1:], data_range=255, size_average=True)
#         msssim=0
#         ssims.append(ssim)
#         msssims.append(msssim)
#         phashs.append(phash)
#     return ssims, msssims, phashs

import piqa

def cal_ssim_dist(video_data, org_videos, video_names, topk=None):
    device = video_data[0].device
    ssims = []
    phashs= []
    for i, (video, org_video) in enumerate(zip(video_data, org_videos)):
        if topk is None:
            phash = stat_phash(org_video.cpu().numpy())
            ssim = torch_ssim((video[:-1]).clamp(0, 1), (video[1:]).clamp(0, 1), data_range=1.0, size_average=True).item()
        else:
            L = int(math.ceil(video.shape[0] * topk))
            phash = stat_phash(org_video.cpu().numpy(), L)
            ssim = (1 - torch_ssim(video[:-1], video[1:], data_range=1.0, size_average=False)).topk(L, largest=True)[0].mean().item()
        ssims.append(ssim)
        # msssims.append(msssim)
        phashs.append(phash)

    return ssims, phashs

def cal_ssim_dist_chunk(video_data, org_videos, video_names, max_frames=200):
    device = video_data[0].device
    ssims = []
    msssims=[]
    psnrs = []
    phashs= []
    for i, (video, org_video) in enumerate(zip(video_data, org_videos)):
        # print(f'calculating ssims for {video_names[i]}')
        num_frames = video.shape[0]
        frame_start = 0
        all_ssims = []
        all_phashs = []
        last_frame = None
        last_frame_org = None

        while frame_start < num_frames:
            frame_end = min(frame_start + max_frames, num_frames)
            video_chunk = video[frame_start:frame_end]
            org_video_chunk = org_video[frame_start:frame_end]
            
            if last_frame is not None:
                extended_video_chunk = torch.cat([last_frame, video_chunk], dim=0)
                extended_org_video_chunk = torch.cat([last_frame_org, org_video_chunk], dim=0)
            else:
                extended_video_chunk = video_chunk
                extended_org_video_chunk = org_video_chunk
            
            phash = stat_phash(extended_org_video_chunk.cpu().numpy(), average=False)
            ssim = torch_ssim(extended_video_chunk[:-1].clamp(0, 1), extended_video_chunk[1:].clamp(0, 1), data_range=1.0, size_average=False).tolist()
            
            all_ssims.extend(ssim)
            all_phashs.extend(phash)
            
            last_frame = video_chunk[-1].unsqueeze(0)
            last_frame_org = org_video_chunk[-1].unsqueeze(0)

            frame_start = frame_end

        ssims.append(np.mean(all_ssims))
        # msssims.append(msssim)
        phashs.append(np.mean(all_phashs))

    return ssims, phashs


# def cal_ssim_dist_chunk(video_data, org_videos, video_names, topk=None, max_frames=100):
#     device = video_data[0].device
#     ssims = []
#     msssims = []
#     phashs = []
#     results = dict()
#     piqa_ssim = piqa.SSIM().to(device)
#     piqa_msssim = piqa.MS_SSIM().to(device)
    
#     for i, (video, org_video) in enumerate(zip(video_data, org_videos)):
#         num_frames = video.shape[0]
#         frame_start = 0
        
#         while frame_start < num_frames:
#             frame_end = min(frame_start + max_frames, num_frames)
#             video_chunk = video[frame_start:frame_end]
#             org_video_chunk = org_video[frame_start:frame_end]
            
#             if topk is None:
#                 phash = stat_phash(org_video_chunk.cpu().numpy())
#                 ssim = piqa_ssim(video_chunk[:-1].clamp(0, 1), video_chunk[1:].clamp(0, 1)).item()
#                 msssim = piqa_msssim(video_chunk[:-1].clamp(0, 1), video_chunk[1:].clamp(0, 1)).item()
#             else:
#                 L = int(math.ceil(video_chunk.shape[0] * topk))
#                 phash = stat_phash(org_video_chunk.cpu().numpy(), L)
#                 ssim = (1 - torch_ssim(video_chunk[:-1], video_chunk[1:], data_range=1.0, size_average=False)).topk(L, largest=True)[0].mean().item()
#                 msssim = 1 - ms_ssim(video_chunk[:-1], video_chunk[1:], data_range=1.0, size_average=False).topk(L, largest=True)[0].mean().item()
            
#             ssims.append(ssim)
#             msssims.append(msssim)
#             phashs.append(phash)
            
#             frame_start = frame_end

#     return ssims, msssims, phashs

if __name__ == "__main__":
    video_folder = '/Volumes/My-Passport/VideoGeneration/normalized_prompt_data/concatenated_dir'
    key_frame_folder = '.temp_key_frame_folder'
    output_csv =  'ssim_inter_frames.csv'
    scores = []
    for file in tqdm(os.listdir(video_folder)):
        if (not file.endswith('.mp4')) or file.startswith('._'):
            continue
        # score = stat_inter_keyframe_ssim(os.path.join(video_folder, file), key_frame_folder)
        ssim_score, psnr, phash, whash = stat_ssim(os.path.join(video_folder, file))
        scores.append({'Video File': os.path.join(video_folder, file), 'ssim': ssim_score, 'psnr': psnr, 'phase': phash, 'whash': whash})
    df = pd.DataFrame(scores)
    df.to_csv(output_csv, index=False)
