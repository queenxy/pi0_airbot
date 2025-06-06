import os
import argparse
import requests
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sketch_via_web.sketch_server import ImageSketchServer

RAW_PATH = '/home/qxy/openpi/data/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='3x3_transfer')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    server = ImageSketchServer()
    server.start()

    name = os.path.join(RAW_PATH, args.name)
    episodes_dir = os.listdir(name)
    for episode in tqdm(episodes_dir, desc=f"Annotating Task {name}"):
        if episode == 'data_recording_info.json' or episode == 'delete.py':
            continue
        episode_dir = f"{name}/{episode}"
        ref_dir = f'{episode_dir}/ref.jpg'
        if os.path.exists(ref_dir):
            continue
        print(int(episode))
        if args.start and int(episode) < args.start:
            continue
        
        wrist_view_dir = f"{episode_dir}/observation.images.cam2.mp4"
        side_view_dir = f"{episode_dir}/observation.images.cam1.mp4"
        # read video frames via cv2
        assert os.path.exists(side_view_dir) and os.path.exists(wrist_view_dir)

        raw_dirs = [side_view_dir]

        cap = cv2.VideoCapture(raw_dirs[0])
        _, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_pil)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ref_imgs = []
        n = 9
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // n * i)
            _, ref_frame = cap.read()
            ref_frame_pil = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            ref_img = Image.fromarray(ref_frame_pil)
            ref_imgs.append(ref_img)

        res, pure_tra, points = server.draw_img(img, ref_imgs)
        res = res.convert('RGB')
        pure_tra = pure_tra.convert('RGBA')
        if res.size == (1280, 960):
            res = res.resize((640, 480), Image.LANCZOS)
        assert res.size == (640, 480)
        assert pure_tra.size == (640, 480)

        res.save(f'{episode_dir}/ref.jpg')
        pure_tra.save(f'{episode_dir}/pure_drawn.png')
        # 将纯轨迹图像叠加到图像上
        combined_img = Image.new('RGB', (640, 480))
        combined_img.paste(res, (0, 0))  # 使用第一个参考图像作为背景
        combined_img.paste(pure_tra, (0, 0), pure_tra)  # 将轨迹透明叠加
        combined_img.save(f'{episode_dir}/ref_with_trajectory.jpg')


        start_time = points[0]['time']
        end_time = points[-1]['time']
        line_duration = end_time - start_time

        # save annotated points
        parsed_points = []
        for i in range(len(points)):
            x = points[i]['x']
            y = points[i]['y']
            t = (points[i]['time'] - start_time) / line_duration
            parsed_points.append([x, y, t])
        parsed_points = np.array(parsed_points)
        np.save(f"{episode_dir}/points.npy", parsed_points)
    

if __name__ == "__main__":
    main()