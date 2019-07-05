# DMPR-PS

This is the implementation of DMPR-PS using PyTorch.

## Requirements

* CUDA
* PyTorch
* OpenCV
* NumPy
* Pillow
* Visdom (optional)
* Matplotlib (optional)

## Pre-trained weights

The [pre-trained weights](https://drive.google.com/open?id=1OuyF8bGttA11-CKJ4Mj3dYAl5q4NL5IT) could be used to reproduce the number in the paper.

## Inference

* Image inference

    ```(shell)
    python inference.py --mode image --detector_weights $DETECTOR_WEIGHTS
    ```

* Video inference

    ```(shell)
    python inference.py --mode video --detector_weights $DETECTOR_WEIGHTS --video $VIDEO
    ```

    `DETECTOR_WEIGHTS` is the trained weights of detector.  
    `VIDEO` is path to the video.  
    View `config.py` for more argument details.

## Prepare data

1. Download ps2.0 from [here](https://cslinzhang.github.io/deepps/), and extract.
2. Download the [labels](https://drive.google.com/open?id=1o6yXxc3RjIs6r01LtwMS_zH91Tk9BFRB), and extract.
3. Perform data preparation and augmentation:

    ```(shell)
    python prepare_dataset.py --dataset trainval --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --output_directory $OUTPUT_DIRECTORY
    python prepare_dataset.py --dataset test --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --output_directory $OUTPUT_DIRECTORY
    ```

    `LABEL_DIRECTORY` is the directory containing json labels.  
    `IMAGE_DIRECTORY` is the directory containing jpg images.  
    `OUTPUT_DIRECTORY` is the directory where output images and labels are.  
    View `prepare_dataset.py` for more argument details.

## Train

```(shell)
python train.py --dataset_directory $TRAIN_DIRECTORY
```

`TRAIN_DIRECTORY` is the train directory generated in data preparation.  
View `config.py` for more argument details (batch size, learning rate, etc).

## Evaluate

* Evaluate directional marking-point detection

    ```(shell)
    python evaluate.py --dataset_directory $TEST_DIRECTORY --detector_weights $DETECTOR_WEIGHTS
    ```

    `TEST_DIRECTORY` is the test directory generated in data preparation.  
    `DETECTOR_WEIGHTS` is the trained weights of detector.  
    View `config.py` for more argument details (batch size, learning rate, etc).

* Evaluate parking-slot detection

    ```(shell)
    python ps_evaluate.py --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --detector_weights $DETECTOR_WEIGHTS
    ```

    `LABEL_DIRECTORY` is the directory containing testing json labels.  
    `IMAGE_DIRECTORY` is the directory containing testing jpg images.  
    `DETECTOR_WEIGHTS` is the trained weights of detector.  
    View `config.py` for more argument details.
