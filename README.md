# Super Resolution of SAR images using GAN
An Implintation of a super resolution for SAR images based on the paper [SAR Image Super-Resolution Based on Noise-Free Generative Adversarial Network](https://ieeexplore.ieee.org/document/8899202).
The dataset used is taken from the [Capella Space Synthetic Aperture Radar (SAR) Open Dataset](https://registry.opendata.aws/capella_opendata/)

## Description
This project contains an implimintation of a super resolution Generial Adverserial Network (GAN) for SAR images.

## Setup
First make sure you have python 3.10.X installed. Then install the enviorment by following one of the following options.
Python VENV:
```bash
python3.10 -m venv superresSAR
source superres/bin/activate
pip install requirements.txt
```

Conda:
```bash
conda create --name superresSAR --python=3.10
conda activate superresSAR
pip install requirements.txt
```

## Training

