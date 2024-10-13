import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pdb

from vbench.utils import load_video, load_dimension_info, dino_transform, dino_transform_Image


def subject_consistency(model, video_path, device, read_frame=False):
    sim = 0.0
    cnt = 0
    video_results = dict()
    if read_frame:
        image_transform = dino_transform_Image(224)
    else:
        image_transform = dino_transform(224)
    # for video_path in tqdm(video_list):
    video_sim = 0.0
    if read_frame:
        video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
        tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
        images = []
        for tmp_path in tmp_paths:
            images.append(image_transform(Image.open(tmp_path)))
    else:
        images = load_video(video_path)
        images = image_transform(images)
    for i in range(len(images)):
        with torch.no_grad():
            image = images[i].unsqueeze(0)
            image = image.to(device)
            image_features = model(image)
            
            image_features = F.normalize(image_features, dim=-1, p=2)
            if i == 0:
                first_image_features = image_features
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
        former_image_features = image_features
    sim += video_sim

    return {os.path.basename(video_path): video_sim/(len(images)-1)}
