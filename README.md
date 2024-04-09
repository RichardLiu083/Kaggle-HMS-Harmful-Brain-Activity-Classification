# Kaggle - HMS-Harmful-Brain-Activity-Classification
**2024/04/09 - Silver Medal - Top 4%**  
**Thanks to my teammate David Jaw**  
[Competition Link](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)

## Solution


## Insight
- Split each soundfile by every 5 seconds, and convert to MelSpectrogram.

## Model
- EfficientNet_b0 with attention module
- EfficientNet_v2s with attention module
- Convnext_base

## Augmentation
- crop size (400, 912)
- RandomBrightnessContrast
- mixup
- HorizontalFlip
- VerticalFlip
- XYMasking
- ShiftScaleRotate

## Training
- use nfnet_l0 to do soft pseudo label
- image size = (400, 912)
- 25 epochs
- lr = 3e-4
- batch size = 16
- mixup prob= 0.3


## Validation
- 5 fold.


## Inference
- 4 notebook ensemble (check ensemble folder)
  > Richard notebook with LB=0.31  
  > David notebook with LB=0.31  
  > Public notebook with LB=0.31  
  > Public notebook with LB=0.29  
