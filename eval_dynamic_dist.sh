export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=1

# video_dir=/root/paddlejob/workspace/env_run/output/liaomingxiang/VideoDatas/VideoDatasets/data_10M
# save_dir=DynamicResultsWebVid10M
video_dir=/root/paddlejob/workspace/env_run/output/liaomingxiang/VideoDatas/generated_videos/Show1
save_dir=Resutls-1
dynamics_save_name=dynamics_results.xlsx
quality_save_name=quality_results.xlsx

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/evaluate_dynamic_ddp_metrics.py \
    --video_dir $video_dir \
    --regress_model_weight_path model_weight/linear_regress_model.pth \
    --save_dir $save_dir --dynamic_score_save_name $dynamics_save_name

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/evaluate_quality_dist.py \
    --video_dir $video_dir \
    --save_dir $save_dir \
    --quality_save_name $quality_save_name \
    --naturalness_path naturalness_results/Show_subset.xlsx

python tools/calculate_metrics.py \
    --video_dir $video_dir \
    --save_dir $save_dir \
    --regress_model_weight_path model_weight/linear_regress_model.pth \
    --dynamic_score_save_name $dynamics_save_name \
    --quality_save_name $quality_save_name --print_detail_quality_results
    # --quality_save_name $quality_save_name