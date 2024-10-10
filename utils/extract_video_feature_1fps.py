import os
import json
import copy
import argparse

from time import sleep
from tqdm import tqdm
from functools import reduce
from subprocess import Popen, PIPE

import torch
import multiprocessing
from threading import Thread


from torch.utils.data import Dataset
from decord import VideoReader
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from LLaVA.llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower


def cut_video(arg):
    rootpath, outpath, name = arg
    video_path = os.path.join(rootpath, name)
    fps = 1
    vid = os.path.join(outpath, name.split(".")[0])
    if os.path.exists(video_path):
        if not os.path.exists(vid):
            os.makedirs(vid)
        os.system(f'ffmpeg -i {video_path} -q 0 -r {fps} {vid}/%06d.jpg')  # -r 5代表每秒抽取5帧。删除该参数即默认全部帧
    else:
        print("File not exists {}".format(video_path))
        return

def run_extract_frames(args):
    print(f'start run_extract')
    gt_questions = json.load(open(args.gt_file))
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    print(f'gt_questions loaded')
    all_videos = []
    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} "):
        if 'video_id' in sample:
            video_name = sample['video_id']
        else:
            video_name = 'v_' + sample['video_name']  # ActivityNet format

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            tmp_name = f"{video_name}{fmt}"
            if os.path.exists(temp_path):
                video_path = temp_path
                video_name = tmp_name
                break
        # Check if the video exists
        if os.path.exists(video_path):
            all_videos.append(video_name)
    
    all_videos_sorted = sorted(set(all_videos))
    print(f'video folder list ready., len={len(all_videos)}, lensorted={len(all_videos_sorted)}')
    # cut_video((args.video_dir, args.frame_dir, all_videos_sorted[0]))
    arg_list = []
    for name in tqdm(all_videos_sorted):
        arg_list.append((args.video_dir, args.frame_dir, name))

    with torch.multiprocessing.Pool() as pool:
        pool.map(cut_video, arg_list)

def get_video_set(param):
    question_list, worker_id = param
    video_set = set()
    video_formats = ['.mp4']
    not_found_set = set()
    for sample in tqdm(question_list, desc=f"worker:{worker_id}"):
        if 'video' in sample:
            video_path = os.path.join(args.video_dir, sample['video'])
            if os.path.exists(video_path):
                video_set.add(sample['video'])
            else:
                not_found_set.add(sample['video'])
        elif 'video_id' in sample:
            finded = False
            for fmt in video_formats:  # Added this line
                video_path = os.path.join(args.video_dir, sample['video_id'] + fmt)
                if os.path.exists(video_path):
                    video_set.add(sample['video_id'] + fmt)
                    finded = True
                    break
            if not finded:
                not_found_set.add(sample['video_id'])

    return video_set, not_found_set

def add(res_a, res_b):
    return res_a[0].union(res_b[0]), res_a[1].union(res_b[1])

def run_scan_video_set(args):
    print(f'start scan')
    gt_questions = json.load(open(args.gt_file))

    print(f'gt_questions loaded')
    video_questions = []
    for sample in tqdm(gt_questions, desc=f"main"):
        if 'video' in sample:
            video_questions.append(sample)
        elif 'video_id' in sample:
            video_questions.append(sample)
    
    print(f'video_questions loaded, total {len(video_questions)}')
    chunk_len = len(video_questions) // args.num_chunks

    param_list = []
    for i in range(0, len(video_questions), chunk_len):
        end = min(i + chunk_len, len(video_questions))
        chunk_list = video_questions[i:end]
        param_list.append((chunk_list, i // chunk_len))

    with torch.multiprocessing.Pool() as pool:
        res_list = pool.map(get_video_set, param_list)
    video_set, not_found_set = reduce(add, res_list)

    print(f'scan video folder {args.gt_file} finished, len={len(video_set)}, notfound={len(not_found_set)}')

    video_list = list(video_set)
    video_list.sort()
    not_found_list = list(not_found_set)
    not_found_list.sort()
    with open(args.output_file, 'w') as f:
        json.dump({'video_list': video_list, 'not_found_list': not_found_list}, f)

def run_convert_video_type(args):

    video_list = json.load(open(args.output_file))['not_found_list']
    video_formats = ['.avi', '.mov', '.mkv', '.webm']
    plist = []
    for video_name in tqdm(video_list):
        real_path = None
        for fmt in video_formats:  # Added this line
            # temp_path = os.path.join(args.video_dir, f"{video_name[:-4]}{fmt}")
            temp_path = os.path.join(args.video_dir, f"{video_name[:]}{fmt}")
            if os.path.exists(temp_path):
                real_path = temp_path
                break
        if real_path is not None:
            out_path = real_path[:-4] + '.mp4'
            if os.path.exists(out_path):
                continue
            ffmpeg_cmd = ['ffmpeg', '-i', real_path, out_path]
            process = Popen(ffmpeg_cmd)
            plist.append(process)
            sleep(1)
    for p in plist:
        p.wait()

class VideoDataset(Dataset):
    def __init__(self, video_dir, video_list, image_processor):
        self.video_dir = video_dir
        self.video_list = video_list
        
        self.processor = image_processor
        self.fps = 1.0

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        video_file = os.path.join(self.video_dir, video_name)
        try:
            vr = VideoReader(video_file, num_threads=4)
            sample_fps = round(vr.get_avg_fps() / self.fps)
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            length = len(frame_idx)
            if length > 700:
                start = (length - 700) // 2
                frame_idx = frame_idx[start:start+700]
            print(f'Loading video, length={len(frame_idx)}, name={video_name}. ', end='')
            video = vr.get_batch(frame_idx).asnumpy()
            images = self.processor.preprocess(video, return_tensors='pt')['pixel_values']
            return images, video_name
        except Exception as e:
            print(f'Error loading video: {e}')
            return "None", video_name

def run_extract_feature(args, video_list):
    args.chunk_idx = 0 #######
    device = torch.device(f"cuda:{args.chunk_idx}")
    torch.cuda.set_device(device)

    vision_tower = CLIPVisionTower(args.vision_tower, args)
    vision_tower = vision_tower.to(device).half()

    dataset = VideoDataset(args.video_dir, video_list, vision_tower.image_processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f'[cuda:{args.chunk_idx}]: dataloader loaded, len={len(dataloader)}')
    thread_list = []
    for batch in tqdm(dataloader, desc=f"cuda:{args.chunk_idx} "):
        try:
            images, video_name = batch
            images = images[0].to(device).half() 
            video_name = video_name[0]
            if images is None:
                continue

            out_path = os.path.join(args.feature_dir, video_name.replace('.mp4', '.safetensors'))

            farther_path = os.path.dirname(out_path)
            if not os.path.exists(farther_path):
                os.makedirs(farther_path)
            with torch.no_grad():
                feature = vision_tower(images)
            feature = feature.cpu().detach().contiguous().half() 

            # save_file({'feature': feature}, out_path)

            save_thread = Thread(target=save_file, args=({'feature': feature}, out_path))
            save_thread.start()
            print(f'Saving video, length={len(feature)}, name={video_name}. ', end='')

            thread_list.append(save_thread)
        
            if len(thread_list) >= args.num_threads:
                print(f'Clearing threads...')
                for thread in thread_list:
                    thread.join()
                thread_list = []

        except Exception as e:
            print(f'Error loading video {video_name}: err={e}')
            continue
    for thread in thread_list:
        thread.join()

def main_extract(args):
    #video_list = json.load(open(args.output_file))['video_list']
    video_list = os.listdir(args.output_file)

    unprocessed_list = []
    for video_name in tqdm(video_list):
        if not os.path.exists(os.path.join(args.feature_dir, video_name.replace('.mp4', '.safetensors'))):
            unprocessed_list.append(video_name)
    video_list = unprocessed_list
    print(f'video_list loaded, len={len(video_list)}')

    if args.num_chunks == 1:
        args.chunk_idx = 0
        args.mm_vision_select_layer = -2
        run_extract_feature(args, video_list)
        return

    chunk_len = len(video_list) // args.num_chunks
    processes = []

    for i in range(args.num_chunks):
        arg = copy.copy(args)
        arg.chunk_idx = i
        arg.mm_vision_select_layer = -2
        video_li = video_list[arg.chunk_idx * chunk_len:(arg.chunk_idx + 1) * chunk_len]
        if i == args.num_chunks - 1:
            video_li = video_list[arg.chunk_idx * chunk_len:]

        process = torch.multiprocessing.Process(target=run_extract_feature, args=(arg, video_li))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print(f"{len(processes)} Processes have finished execution.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--frame_dir', help='Directory to write pickle files.', required=False)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=False)
    parser.add_argument('--output_file', help='Path to video list file.', required=False)
    parser.add_argument('--feature_dir', help='Path to the video feature.', required=False)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--cmd", type=str, choices=['scan', 'extract', 'convert'])

    parser.add_argument("--vision_tower", type=str, default='openai/clip-vit-large-patch14-336')

    args = parser.parse_args()

    # run_extract_frames(args)

    if args.cmd == 'scan':
        run_scan_video_set(args)
    elif args.cmd == 'convert':
        run_convert_video_type(args)
    elif args.cmd == 'extract':
        torch.multiprocessing.set_start_method('spawn')
        main_extract(args)
    
