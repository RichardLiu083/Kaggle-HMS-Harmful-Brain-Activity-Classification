# Kaggle - HMS-Harmful-Brain-Activity-Classification
**2024/04/09 - Silver Medal - Top 4%**  
**Thanks to my teammate - David Jaw**  
[Competition Link](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
![image](https://github.com/RichardLiu083/Kaggle-HMS-Harmful-Brain-Activity-Classification/blob/master/kaggle.PNG)

## Solution


## Insight
- Convert raw eeg data to spectrogram with time length 10s/30s.
- Combine eeg_spectrogram with kaggle_spectrogram into shape (400, N) image, N depands on time length .
- Use attention module before Avgpooling in efficientnet to learn time series feature.
- Using 2 stage training, 1st stage with all data, 2nd stage with only voter>7 data. (high quality label data)
- Doing pseudo label to voter<7 data, then add to 2nd stage training. (replace low quality data with pseudo label)
- Multimodal is crucial, so combine public notebook 1D raw eeg model with our 2D spectrogram model.

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
- image size = (400, 912)
- 25 epochs with 1st stage, 15 epoch with 2nd stage.
- lr = 3e-4
- batch size = 16
- mixup prob= 0.3

## Inference
- 4 notebook ensemble (check Ensemble_Notebook folder)
  - Richard notebook with LB=0.31  
  - David notebook with LB=0.31  
  - Public notebook with LB=0.31  
  - Public notebook with LB=0.29  
