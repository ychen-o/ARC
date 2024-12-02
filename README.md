## Requirements

Ensure the following libraries are installed:

- Python 3.x
- pandas==1.5.3
- numpy==1.24.4
- feather-format==0.4.1
- feather==0.1.2
- matplotlib==3.7.1
- scipy==1.9.1
- tqdm==4.66.4
- opencv-python==4.7.0.72
- opencv_python_headless==4.10.0.84
- torch==1.13.1+cu117
- torchvision==0.14.1+cu117
- ultralytics==8.2.51

## Datasets
The original video dataset for ARC comes from a previous paper and can be accessed [<u>here</u>](https://github.com/stanford-futuredata/blazeit). Additionally, we provide pre-computed labels in .npy or .csv format, and intermediate experimental data (e.g., the execution speed of the Mask R-CNN model on our hardware) in the form of configuration files so that you can quickly reproduce experiments in ARC.
 

## Reproducing Experiments
Navigate to the `experiments` directory and run `experiment_main.py` to reproduce the key results presented in ARC. The `main` function will execute a series of experiments in sequence according to the provided configuration order. The results will be saved in the `experiments/result` directory.


