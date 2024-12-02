import os

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIRS = {
    'video': os.path.join(BASE_DIR, 'data', 'video'),
    'output_yolo': os.path.join(BASE_DIR, 'data', 'YOLOv5s'),
    'output_mask': os.path.join(BASE_DIR, 'data', 'MaskRCNN'),
    'output_cdf': os.path.join(BASE_DIR, 'data', 'CDF'),
    'everest': os.path.join(BASE_DIR, 'data', 'Everest'),
}

# Map the video file to the category index based on YOLOv5, which will be converted to adapt to Mask R-CNN during formal processing
VIDEO_CLASS_MAP = {        
    'venice-rialto.mp4': [8],  
    'venice-grand-canal.mp4': [8],  
    'taipei-hires.mp4': [2, 5, 7],
    'amsterdam.mp4': [2, 5, 7],
    'jackson-town-square.mp4': [2, 5, 7],
}
identify_objects=True

# Map the video file to the maximum number of objects
VIDEO_MAX_SCORE_DIRS = {
    'venice-rialto':13, 
    'venice-grand-canal':16, 
    'taipei-hires':19, 
    'amsterdam':7,  
    'jackson-town-square':10, 
    'taipei-hires-car':18, 
    'taipei-hires-truck':7, 
    'taipei-hires-bus':7
}