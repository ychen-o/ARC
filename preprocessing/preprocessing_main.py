from video_processor import YOLOProcessor, MaskRCNNProcessor
import torchvision
from ultralytics import YOLO
from preprocessing_config import DATA_DIRS,VIDEO_CLASS_MAP,identify_objects,VIDEO_MAX_SCORE_DIRS 
from data_loader import load_data, load_model_results
from cdf_generator import gen_cdf
import os
import torch
import numpy as np
import pandas as pd

def prepare_data():
    categorys={2:'car',5:'bus',7:'truck',8:'boat'}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    yolo_model = YOLO('model_weights/yolov5su.pt')
    mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    mask_model.eval()

    for video_file, classes in VIDEO_CLASS_MAP.items():
        video_path = os.path.join(DATA_DIRS['video'], video_file)
        if os.path.exists(video_path):
            yolo_processor = YOLOProcessor(yolo_model, device, DATA_DIRS['video'], DATA_DIRS['output_yolo'],identify_objects=identify_objects)
            mask_processor = MaskRCNNProcessor(mask_model, device , DATA_DIRS['video'], DATA_DIRS['output_mask'],identify_objects=identify_objects)

            yolo_results = yolo_processor.process_video(video_file,classes)
            mask_results = mask_processor.process_video(video_file,[c+1 for c in classes])

            if identify_objects:
                for category in classes:
                    frame_counts = [(frame_idx, count) for (frame_idx, class_id), count in yolo_results.items() if class_id == category]
                    df_yolo = pd.DataFrame(frame_counts, columns=['Frame Index', 'Count']).sort_values('Frame Index')
                    output_path = os.path.join(DATA_DIRS['output_yolo'], f'{os.path.splitext(video_file)[0]}-{categorys[category]}.npy')
                    np.save(output_path, df_yolo['Count'].to_numpy())
                for category in [c+1 for c in classes]:
                    frame_counts = [(frame_idx, count) for (frame_idx, class_id), count in mask_results.items() if class_id == category]
                    df_mask = pd.DataFrame(frame_counts, columns=['Frame Index', 'Count']).sort_values('Frame Index')
                    output_path = os.path.join(DATA_DIRS['output_mask'], f'{os.path.splitext(video_file)[0]}-{categorys[category-1]}.npy')
                    np.save(output_path, df_mask['Count'].to_numpy())
    
            frame_counts_total = [(frame_idx, count) for (frame_idx, class_id), count in yolo_results.items() if class_id == -1]
            df_yolo_total = pd.DataFrame(frame_counts_total, columns=['Frame Index', 'Count']).sort_values('Frame Index')
            output_path = os.path.join(DATA_DIRS['output_yolo'], f'{os.path.splitext(video_file)[0]}.npy')
            np.save(output_path, df_yolo_total['Count'].to_numpy())
            
            frame_counts_total = [(frame_idx, count) for (frame_idx, class_id), count in mask_results.items() if class_id == -1]
            df_mask_total = pd.DataFrame(frame_counts_total, columns=['Frame Index', 'Count']).sort_values('Frame Index')
            output_path = os.path.join(DATA_DIRS['output_mask'], f'{os.path.splitext(video_file)[0]}.npy')
            np.save(output_path, df_mask_total['Count'].to_numpy())
        else:
            print(f"Warning: {video_file} not found in video directory.")

def prepare_cdf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for directory, max_score in VIDEO_MAX_SCORE_DIRS.items():
        pi = load_data('everest', f'{directory}/pi.npy')
        sigma = load_data('everest', f'{directory}/sigma.npy')
        mu = load_data('everest', f'{directory}/mu.npy')
        remained_data = load_data('everest', f'{directory}/remained.npy')[:, 0]
        predicates = load_model_results('output_mask', directory)
        yolov5s = load_model_results('output_yolo', directory)

        cdf = gen_cdf(pi, mu, sigma, max_score, device)
        CDFs = np.zeros((len(remained_data), max_score), dtype=object)
        for i in range(len(remained_data)):
            CDFs[i] = cdf[i].tolist()
        
        df = pd.DataFrame(CDFs.tolist(), columns=[f'{i}' for i in range(max_score)])
        df['predicates'] = predicates[remained_data]
        df['yolov5s'] = yolov5s[remained_data]
        df.to_csv(os.path.join(DATA_DIRS['output_cdf'], f'{directory}.csv'), float_format='%.10f', index=False)
 