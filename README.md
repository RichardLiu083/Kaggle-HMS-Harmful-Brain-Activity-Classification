# Kaggle - HMS-Harmful-Brain-Activity-Classification
**2024/04/09 - Silver Medal - Top 4%**  
**Thanks to my teammate - [David Jaw](https://github.com/davidjaw)**  
[Competition Link](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
![image](https://github.com/RichardLiu083/Kaggle-HMS-Harmful-Brain-Activity-Classification/blob/master/kaggle.PNG)

## Insight
- Convert raw eeg data to spectrogram with time length 10s/30s.
- Combine eeg_spectrogram with kaggle_spectrogram into shape (400, N) image, N depands on time length .
- Use attention module before Avgpooling in efficientnet to learn time series feature.
- Using 2 stage training, 1st stage with all data, 2nd stage with only voter>7 data. (high quality label data)
- Doing pseudo label to voter<7 data, then add to 2nd stage training. (replace low quality data with pseudo label)
- Multimodal is crucial, so combine public notebook 1D raw eeg model with our 2D spectrogram model.

## Model
- **Richard Pipeline:**
  - EfficientNet_b0 with attention module
  - EfficientNet_v2s with attention module
  - Convnext_base \n
- **David Pipeline:**
  - EfficientNet_b0
  - MobileNetv3-Large

## Augmentation
- **Richard Pipeline:**
  - crop size (400, 912)
  - RandomBrightnessContrast
  - mixup
  - HorizontalFlip
  - VerticalFlip
  - XYMasking
  - ShiftScaleRotate
- **David Pipeline:**
  - crop size (400, 256 + 512)
  - RandomBrightnessContrast
  - mixup
  - HorizontalFlip
  - YMasking

## Training
- Richard Pipeline:
  - image size = (400, 912)
  - 25 epochs with 1st stage, 15 epoch with 2nd stage.
  - lr = 3e-4
  - batch size = 16
  - mixup prob= 0.3

  
- David pipeline:
  - image size = (400, 256 + 512)
    - As discussed [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010), I get the spectrogram and eeg clips with respective offsets. 
    - For spectrogram provided by Kaggle, the clip shape is around `[400, 300]`, and I randomly crop it by width of `256` during training, pick the first `256` during testing.
    - For eeg clips, I sampled them with time length of `10s/15s/30s/40s` with `n_mel` set at 50 and `hop_length=len(x) // 256`, i.e., each eeg spectrogram is at shape `[50, 256]`. Finally, I concat them into a `[400, 512]`.
  - 3 epochs with 1st stage, 12 epochs with 2nd stage. (I used votes of 10 as the threshold)
  - lr = 2e-4 with [Lion](https://github.com/lucidrains/lion-pytorch) optimizer
  - batch size = 32
  - mixup prob = 0.3

## Inference
- 4 notebook ensemble (check Ensemble_Notebook folder)
  - Richard notebook with LB=0.31  
  - David notebook with LB=0.31  
  - Public notebook with LB=0.31  (tensorflow pipeline)
  - Public notebook with LB=0.29  (public notebook ensemble)

- final result LB=0.28
