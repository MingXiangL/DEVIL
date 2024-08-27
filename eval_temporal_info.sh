export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS eval_dynamic_ddp.py \
#     --video_dir /root/paddlejob/workspace/env_run/output/liaomingxiang/Didemo/videos-1 \
#     --annotations_file /root/paddlejob/workspace/env_run/output/liaomingxiang/Didemo/test_a.json \
#     --save_dir DynamicResultsDidemoTest


# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS eval_dynamic_ddp.py \
#     --video_dir /root/paddlejob/workspace/env_run/output/liaomingxiang/MSRVTT/MSRVTT_Videos \
#     --annotations_file /root/paddlejob/workspace/env_run/output/liaomingxiang/MSRVTT/msrvtt_merged_data.json \
#     --save_dir DynamicResultsMSRVTT


# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS eval_dyanmic_ddp_metrices.py \
#     --video_dir /root/paddlejob/workspace/env_run/output/luhannan/Codes/gen_video_collections/concatenated_dir-1/concatenated_dir \
#     --save_dir DynamicResultsGeneratedVideos

# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS eval_dyanmic_ddp_metrices.py \
#     --video_dir /root/paddlejob/workspace/env_run/output/liaomingxiang/DinaBench/videos_test \
#     --save_dir DynamicResultsGeneratedVideos-1


python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 12345 evaluate_tempora_info_ddp_metrices.py \
    --video_dir /home/LiaoMingxiang/Workspace2/candidate_videos \
    --save_dir DynamicResultsGeneratedVideos-temporal-info

# vc2_person_very_high_0006.mp4