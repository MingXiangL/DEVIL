#!/bin/bash

# input_video=$1
# output_folder=$2
# mkdir -p "$output_folder"

extract_and_combine_frames() {
    local VIDEO_FILE="$1"
    local OUTPUT_DIR="$2"
    local OUTPUT_FRAMES="$3"
    local OUTPUT_VIDEO="$4"

    # 确保输出目录存在
    mkdir -p "$OUTPUT_DIR"
    # 获取总帧数
    local TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$VIDEO_FILE")
    # 判断总帧数是否小于预期提取的帧数
    if [ "$TOTAL_FRAMES" -le "$OUTPUT_FRAMES" ]; then
        # 总帧数少于或等于要提取的帧数，提取所有帧
        ffmpeg -i "$VIDEO_FILE" -vsync 0 "$OUTPUT_DIR/frame_%03d.png" -y -loglevel error 
    else
        # 计算间隔
        local INTERVAL=$(expr $TOTAL_FRAMES / $OUTPUT_FRAMES)
        
        # 提取帧
        ffmpeg -i "$VIDEO_FILE" -vf "select='not(mod(n,$INTERVAL))',setpts=N/(FRAME_RATE*TB)" -vsync vfr "$OUTPUT_DIR/frame_%03d.png" -y -loglevel error 
    fi

    # 创建新视频
    ffmpeg -framerate 24 -i "$OUTPUT_DIR/frame_%03d.png" -c:v libx264 -pix_fmt yuv420p -r 24 "$OUTPUT_VIDEO" -y -loglevel error 
    rm -rf $OUTPUT_DIR/*.png
}

calculate_temporal_info() {
    local input_video=$1    
    local output_folder=$2
    local sigma=${3:-5.0}
    local num_frames=${4:-128}
    if [ -z "$output_folder" ]; then
        echo "Error: output_folder is empty."
        exit 1
    fi
    if [ -z "$input_video" ]; then
        echo "Error: output_folder is empty."
        exit 1
    fi

    local extracted_video=${output_folder}/extracted_video.mp4
    local filtered_video=${output_folder}/filtered_video.mp4
    local down_scaled_video=${output_folder}/down_scaled_video.mp4

    mkdir -p $output_folder
    local total_frames=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$input_video")
    
    # extract_and_combine_frames $input_video $output_folder $num_frames $extracted_video
    ffmpeg -i $input_video -vf "scale=640:480" $down_scaled_video -y -loglevel error 

    ffmpeg -i $down_scaled_video -vf "gblur=sigma=$sigma" $filtered_video -y -loglevel error 

    # 对视频进行压缩
    ffmpeg -i $filtered_video -c:v libx264 -y $output_folder/output_spatial_temporal.mp4 -loglevel error -g $total_frames
    # 计算视频的熵H(f1,f1,...,fn)
    total_video_size_encoded=$(echo "scale=4; $(wc -c < $output_folder/output_spatial_temporal.mp4)/1" | bc)

    # 提取关键帧
    ffmpeg -i $filtered_video -vf "select=eq(pict_type\,I)" -vsync vfr $output_folder/output_%03d.png -y -loglevel error  -g $total_frames
    # 关键帧变成视频
    ffmpeg -i $output_folder/output_%03d.png  -c:v libx264 $output_folder/output_key_1.mp4 -y -loglevel error
    

    # 每个关键帧各自变成视频，并计算中大小sum_i(H(fki))
    key_frame_total_size=0.0
    for img in $output_folder/output_*.png; do
        ffmpeg -i "$img" -c:v libx264 "${img%.png}.mp4" -y -loglevel error 
        size=$(echo "scale=4; $(wc -c < ${img%.png}.mp4)/1" | bc)
        key_frame_total_size=$(echo "scale=4; $key_frame_total_size + $size" | bc)
    done

    # 提取第一帧
    ffmpeg -i $filtered_video -frames:v 1 first_frame.png -y -loglevel error 
    ffmpeg -i first_frame.png -c:v libx264 $output_folder/first_frame_video.mp4 -y -loglevel error 
    # 第一帧的熵H(f1)
    first_frame_size=$(echo "scale=4; $(wc -c < $output_folder/first_frame_video.mp4)/1" | bc)

    # 关键帧的熵H(fk1,fk2,...,fkm)
    key_frame_size_encoded=$(echo "scale=4; $(wc -c < $output_folder/output_key_1.mp4)/1" | bc)

    # echo total_video_size_encoded: $total_video_size_encoded
    # echo first_frame_size: $first_frame_size
    # echo key_frame_size_encoded: $key_frame_size_encoded
    # echo key_frame_total_size $key_frame_total_size

    local temporal_info=$(echo "scale=4; $total_video_size_encoded + $key_frame_size_encoded - $key_frame_total_size - $first_frame_size" | bc)
    echo $temporal_info
    rm -rf $output_folder
}

cal_temporal_info_batch(){
    local prefix=$1
    local sigma=${2:-5}
    local output_folder_1=temp_info_sigma_${sigma}_$(basename $prefix)
    local output_folder=$output_folder_1/key_frames
    
    
    mkdir -p $output_folder
    mkdir -p $output_folder_1
    rm -rf $output_folder/*.png
    rm -rf $output_folder_1/*.png

    local total_folders=$(find "$prefix" -mindepth 1 -maxdepth 1 -type d | wc -l)
    local current_folder=1

    output_file1=${output_folder_1}/temporal_info.csv
    echo "" > $output_file1
    for video_folder in "${prefix}"/*;
    do
        output_file=${video_folder}/temporal_info.csv

        ####### 进度条 #######
        percent=$(( 100 * current_folder / total_folders ))
        bar=$(printf '%*s' "$(( percent / 2 ))" '' | tr ' ' '=')
        printf "\rProgress: [%-50s] %d%%" "$bar" "$percent"
        let current_folder++
        ####### 进度条 #######

        # # 清空输出文件
        echo "" > $output_file
        for video_file in $video_folder/*.mp4; do
            tempora_info=$(calculate_temporal_info "$video_file" "$output_folder" "$sigma")
            echo "$(basename $video_file),$tempora_info" >> $output_file
            echo "$(basename $video_file),$tempora_info" >> $output_file1
        done
    done
}

cal_temporal_info_single(){
    local prefix=$1
    local sigma=${2:-5}
    local output_folder_1=temp_info_sigma_${sigma}_$(basename $prefix)
    local output_folder=$output_folder_1/key_frames
    
    rm -rf $output_folder
    rm -rf $output_folder_1

    mkdir -p $output_folder
    mkdir -p $output_folder_1

    local total_files=$(find "$prefix" -type f -name "*.mp4" | wc -l)
    local current_file=1

    output_file=${output_folder_1}/temporal_info.csv
    echo "" > $output_file
    for video_file in "${prefix}"/*.mp4;
    do
        # video_file=${prefix}/fn_lavie_bicycle_low_0001.mp4
        ####### 进度条 #######
        percent=$(( 100 * current_file / total_files ))
        bar=$(printf '%*s' "$(( percent ))" '' | tr ' ' '=')
        printf "\rProgress: [%-50s] %d%%" "$bar" "$percent"
        let current_file++
        ####### 进度条 #######

        # # 清空输出文件
        
        tempora_info=$(calculate_temporal_info "$video_file" "$output_folder" "$sigma")
        echo "$(basename $video_file),$tempora_info" >> $output_file
    done
}


# cal_temporal_info_batch /Volumes/My-Passport/Download-back/Downloads/vidprom_obj_cls20_1_rename_2/lavie_obj $sigma
# cal_temporal_info_batch /Volumes/My-Passport/Download-back/Downloads/vidprom_obj_cls20_1_rename_2/fn_lavie_obj $sigma
# cal_temporal_info_batch /Volumes/My-Passport/Download-back/Downloads/vidprom_obj_cls20_1_rename_2/fn_vc1_obj $sigma
# cal_temporal_info_batch /Volumes/My-Passport/Download-back/Downloads/vidprom_obj_cls20_1_rename_2/vc1_obj $sigma
cal_temporal_info_single /root/paddlejob/workspace/env_run/output/liaomingxiang/cal_dynamic_code/generated_videos/concatenated_dir