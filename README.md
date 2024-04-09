# Kaggle - BirdCLEF 2023 - Identify bird calls in soundscapes
**2023/05/25 - Silver Medal - Top 5%**  
[Competition Link](https://www.kaggle.com/competitions/birdclef-2023)
![image](https://github.com/RichardLiu083/Kaggle-BirdCLEF-2023/blob/main/birdcall_rank.png)

## Solution


## Insight
- Split each soundfile by every 5 seconds, and convert to MelSpectrogram.
- keep the aspect ratio of MelSpectrogram image is important.
- Horizontal、Vertical flip will lead to performence drop. (since I use MelSpectrogram as input)
- Use external dataset to build pretrained model. (previous birdcall competition)
- GroupKFold (K=5) split by bird type.
- Train each model for 5 fold, then use model soup to do combination.
- Since there are some images without signal (label in soundfile level), so use soft-pseudo label for OOF, and repeat until no LB gain.
- Use (hard label * 0.5 + soft pseudo label * 0.5) as new label for each new cycle training.
- if number of samples in one species > 500, random choose 500；if < 50, random copy choose 50.

## Model
- EfficientNet_b0
- EfficientNet_v2s
- Convnext_v2_base

## Augmentation
- SmallestMaxSize
- RandomBrightnessContrast
- mixup
- GaussNoise
- Cutout
- CoarseDropout
- ShiftScaleRotate

## Training
- use nfnet_l0 to do soft pseudo label
- image size = 128
- 50 epochs
- lr = 5e-4
- batch size = 64
- mixup prob= 0.65


## Validation
- 5 fold.
- if number of species < 5, put it into training dataset.
- CMAP score have high correlation with LB score.


## Inference
- 3 models weighted ensemble.
