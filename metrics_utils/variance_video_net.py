import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from typing import List
import argparse
import pdb
import os
import tqdm
from PIL import Image
import torch.multiprocessing as mp
import torch.nn.functional as F
from decord import VideoReader, gpu
import pandas as pd

# 初始化模型和预处理流程
def initialize_model(args):
    if args.model == 'resnet':
        res3d = models.video.r2plus1d_18(pretrained=True)
        res3d.cuda()
        res3d.eval()  # 设置为评估模式
    elif args.model == 'swin':
        res3d = models.video.swin3d_t(pretrained=True)
        res3d.cuda()
    else:
        raise NotImplementedError
    return res3d

def initialize_transform():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])
    return transform

# 加载和预处理单个视频
# def load_video(filename: str, transform):
#     cap = cv2.VideoCapture(filename)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB
#         frame = transform(Image.fromarray(frame))  # 应用预处理
#         frames.append(frame)
#     cap.release()
#     if len(frames) == 0:
#         return None
#     return torch.stack(frames).unsqueeze(0)  # 增加一个批次维度
def load_video(filename: str, transform, rank):
    vr = VideoReader(filename)
    frames = []
    for frame in vr:
        frame = transform(Image.fromarray(frame.asnumpy()))  # 应用预处理
        frames.append(frame)
    if len(frames) == 0:
        return None
    return torch.stack(frames).unsqueeze(0)  # 增加一个批次维度


# 批量处理视频并提取特征
def extract_features(file, model, transform, device, rank):
    video = load_video(file, transform, rank).permute(0,2,1,3,4).to(device)
    # video_batch = torch.cat(video, dim=0)  # 沿批次维度合并
    with torch.no_grad():
        features = model(video)
    return features.flatten(-2,-1).mean(-1)

def cal_variance(features):
    features_mean = features.mean(axis=0, keepdim=True)
    # features_normed = features - features_mean
    dist = (1 - F.cosine_similarity(features_mean[:, None], features[None], dim=-1)).mean(1)
    return dist.item()

def run_inference(rank, args, seq_queue, output_queue):
    device = torch.device('cuda', rank)
    model = initialize_model(args).to(device)
    processing =  initialize_transform()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    while True:
        file = seq_queue.get()
        if file == 'END':
            break
        features = extract_features(file, feature_extractor, processing, device, rank)[0].permute(1,0)
        variance = cal_variance(features)
        output_queue.put({'video_path':file, 'dist':variance})
        print(f'{rank}: {file} done')
            

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from video frames using DINO or CLIP models.")
    parser.add_argument("--video_folder", type=str, default='/root/paddlejob/workspace/env_run/output/luhannan/Codes/testset/vidprom_obj_cls20_1_rename_2', help="Path to the video file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing frames.")
    parser.add_argument("--model", type=str, default='resnet', choices=['resnet', 'clip', 'swin'], help="Model to use for feature extraction.")
    parser.add_argument("--output_file", type=str, default='res3d_score.csv', help="Path to the output CSV file.")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use.")
    
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)

    video_paths = [os.path.join(args.video_folder, f) for f in os.listdir(args.video_folder)]
    video_paths = sorted(video_paths)

    seq_queue = mp.Queue()
    output_queue = mp.Queue()
    for vname in video_paths:
        # prompt_path = os.path.join(prompt_dir, fname)
        seq_queue.put(vname)
    for _ in range(args.n_gpu):
        seq_queue.put('END')

    model = initialize_model(args)
    transform = initialize_transform()
    mp.spawn(run_inference, nprocs=args.n_gpu, args=(args, seq_queue, output_queue))

    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    pd.DataFrame(results).to_csv(args.output_file, index=False)
