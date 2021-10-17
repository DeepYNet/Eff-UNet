# Eff-UNet

This repository contains EffUnet segmentation model. The model files and training files are uploaded. 

## Model structure

![EffUnet](https://github.com/DeepYNet/Eff-UNet/blob/main/assets/EffUnet.PNG)

## Steps to Run
- Clone the repo
- Activate conda environment
- In train.py change the crops and diffs directory path.
- `python train.py`

## Thing to try
- [x] Padding vs interpolation. We are going with 310x250 for now.
- [ ] Pass logit features to the decoder i.e the features before normalization.
- [ ] Try DANet for deepfakes classification.
