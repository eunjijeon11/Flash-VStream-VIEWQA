import os

video_dir_path = "/mnt/hdd2/VIEWQA/VIEWQA_total_resized_224"
all_videos = os.listdir(video_dir_path)

tensors_dir_path = "data/finetune/video_features"
all_features = os.listdir(tensors_dir_path)

if (len(all_videos) == len(all_features)):
	print("All videos have been processed.")