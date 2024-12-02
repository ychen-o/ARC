import torch
import cv2
import numpy as np
from tqdm import tqdm
import time
from torchvision.transforms import functional as F
import os

class VideoProcessor:
    def __init__(self, model, device,  video_folder, output_folder, identify_objects=False):
        self.model = model.to(device)
        self.device = device
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.identify_objects = identify_objects

    def process_video(self, video_file):
        raise NotImplementedError("Subclasses should implement this method")

class YOLOProcessor(VideoProcessor):
    def __init__(self, model, device,  video_folder, output_folder, threshold=0.5, **kwargs):
        super().__init__(model, device, video_folder, output_folder, **kwargs)
        self.threshold = threshold

    def process_video(self, video_file, class_ids):
        cap = cv2.VideoCapture(os.path.join(self.video_folder, video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results_summary = {}
        with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                start_time = time.time()
                result = self.model([frame], verbose=False)[0]
                end_time = time.time()
                pbar.set_postfix(time=f"{end_time - start_time:.4f} sec/frame")             
                results_summary[(frame_idx, -1)] = 0
                for cid in class_ids:
                    results_summary[(frame_idx , cid)] = 0
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)       
                    if class_id in class_ids and float(box.conf) > self.threshold:
                        results_summary[(frame_idx, class_id)] += 1
                total_count = sum([results_summary[(frame_idx, cid)] for cid in class_ids])
                results_summary[(frame_idx , -1)] = total_count
                    
                pbar.update(1)
        cap.release()
        return results_summary

class MaskRCNNProcessor(VideoProcessor):
    def __init__(self, model, device, video_folder, output_folder, score_threshold=0.9, **kwargs):
        super().__init__(model, device, video_folder, output_folder, **kwargs)
        self.score_threshold = score_threshold

    def process_video(self, video_file, class_ids):
        cap = cv2.VideoCapture(os.path.join(self.video_folder, video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results_summary = {}
        times = []
        with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar: 
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_frame = F.to_tensor(frame_rgb).unsqueeze(0).to(self.device)      
                start_time = time.time()
                with torch.no_grad():
                    output = self.model(tensor_frame)
                end_time = time.time()
                times.append(end_time - start_time)
                pbar.set_postfix(time=f"{np.mean(times):.4f} sec/frame")
                pred_classes = output[0]['labels'].cpu().numpy()
                pred_scores = output[0]['scores'].cpu().numpy()
                filtered_classes = pred_classes[pred_scores > self.score_threshold]
                for c_i in class_ids:
                    results_summary[(frame_idx , c_i)] = 0
                for class_id in filtered_classes:
                    if class_id in class_ids:
                        results_summary[(frame_idx , class_id)] += 1
                total_count = sum([results_summary[(frame_idx, cid)] for cid in class_ids])
                results_summary[(frame_idx , -1)] = total_count
                del frame,output
                
                pbar.update(1)
        cap.release()
        return results_summary
