# DMPR-PS

This is the implementation of [DMPR-PS](https://github.com/Teoge/DMPR-PS/blob/master/DMPR-PS.pdf) using PyTorch.

## Requirements

* PyTorch
* CUDA (optional)
* Other requirements  
    `pip install -r requirements.txt`

## Pre-trained weights

The [pre-trained weights](https://drive.google.com/open?id=1OuyF8bGttA11-CKJ4Mj3dYAl5q4NL5IT) could be used to reproduce the number in the paper.

## Inference

* Image inference

    ```(shell)
    python inference.py --mode image --detector_weights $DETECTOR_WEIGHTS --inference_slot
    ```

* Video inference

    ```(shell)
    python inference.py --mode video --detector_weights $DETECTOR_WEIGHTS --video $VIDEO --inference_slot
    ```

    Argument `DETECTOR_WEIGHTS` is the trained weights of detector.  
    Argument `VIDEO` is path to the video.  
    View `config.py` for more argument details.

## Prepare data

1. Download ps2.0 from [here](https://cslinzhang.github.io/deepps/), and extract.
2. Download the [labels](https://drive.google.com/open?id=1o6yXxc3RjIs6r01LtwMS_zH91Tk9BFRB), and extract.  
(In case you want to label your own data, you can use [`directional_point` branch of my labeling tool **MarkToolForParkingLotPoint**](https://github.com/Teoge/MarkToolForParkingLotPoint/tree/directional_point).)
3. Perform data preparation and augmentation:

    ```(shell)
    python prepare_dataset.py --dataset trainval --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --output_directory $OUTPUT_DIRECTORY
    python prepare_dataset.py --dataset test --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --output_directory $OUTPUT_DIRECTORY
    ```

    Argument `LABEL_DIRECTORY` is the directory containing json labels.  
    Argument `IMAGE_DIRECTORY` is the directory containing jpg images.  
    Argument `OUTPUT_DIRECTORY` is the directory where output images and labels are.  
    View `prepare_dataset.py` for more argument details.

## Train

```(shell)
python train.py --dataset_directory $TRAIN_DIRECTORY
```

Argument `TRAIN_DIRECTORY` is the train directory generated in data preparation.  
View `config.py` for more argument details (batch size, learning rate, etc).

## Evaluate

* Evaluate directional marking-point detection

    ```(shell)
    python evaluate.py --dataset_directory $TEST_DIRECTORY --detector_weights $DETECTOR_WEIGHTS
    ```

    Argument `TEST_DIRECTORY` is the test directory generated in data preparation.  
    Argument `DETECTOR_WEIGHTS` is the trained weights of detector.  
    View `config.py` for more argument details (batch size, learning rate, etc).

* Evaluate parking-slot detection

    ```(shell)
    python ps_evaluate.py --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --detector_weights $DETECTOR_WEIGHTS
    ```

    Argument `LABEL_DIRECTORY` is the directory containing testing json labels.  
    Argument `IMAGE_DIRECTORY` is the directory containing testing jpg images.  
    Argument `DETECTOR_WEIGHTS` is the trained weights of detector.  
    View `config.py` for more argument details.

## Citing DMPR-PS

If you find DMPR-PS useful in your research, please consider citing:

```()
@inproceedings{DMPR-PS,
Author = {Junhao Huang and Lin Zhang and Ying Shen and Huijuan Zhang and Shengjie Zhao and Yukai Yang},
Booktitle = {2019 IEEE International Conference on Multimedia and Expo (ICME)},
Title = {{DMPR-PS}: A novel approach for parking-slot detection using directional marking-point regression},
Month = {Jul.},
Year = {2019},
Pages = {212-217}
}
```
