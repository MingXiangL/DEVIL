#!/bin/bash

video_dir=""
gemini_api_key=""
num_gpus=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --video_dir)
            video_dir="$2"
            shift 2
            ;;
        --gemini_api_key)
            gemini_api_key="$2"
            shift 2
            ;;
        --num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ -z "$video_dir" ] || [ -z "$gemini_api_key" ]; then
    echo "Usage: $0 --video_dir <dir_to_your_videos> --gemini_api_key <your_gemini_api_key> [--num_gpus <number_of_gpus>]"
    echo "Note: If not specified, num_gpus defaults to 1"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

save_dir=Devil-eval-results-$(date +"%Y%m%d_%H%M%S")
dynamics_save_name=dynamics_results.xlsx
quality_save_name=quality_results.xlsx


# Calculate dynamics score of each video
python -m torch.distributed.launch --nproc_per_node=$num_gpus tools/evaluate_dynamic_ddp_metrics.py \
    --video_dir $video_dir \
    --save_dir $save_dir --dynamic_score_save_name $dynamics_save_name

# Calculate quality score of each video
python -m torch.distributed.launch --nproc_per_node=$num_gpus tools/evaluate_quality_dist.py \
    --video_dir $video_dir \
    --save_dir $save_dir \
    --quality_save_name $quality_save_name \
    --gemini_api_key $gemini_api_key

# Calculate model metric of the model
python tools/calculate_metrics.py \
    --video_dir $video_dir \
    --save_dir $save_dir \
    --dynamic_score_save_name $dynamics_save_name \
    --quality_save_name $quality_save_name --print_detail_quality_results
