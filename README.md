# ResNet-34 Model for CIFAR-10 Image Classification

This project implements a PyTorch-based ResNet-34 model for image classification on the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes. The model outputs class probabilities for each input image.

# Navigation
- [Research Paper - Deep Residual Learning for Image Recognition](#research-paper)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
- [Performance](#performance)

# Research Paper - Deep Residual Learning for Image Recognition <a id="research-paper"></a>

This model is a ResNet-34 deep-learning architecture proposed in the 2015 research paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), authored by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. The implementation is adapted for CIFAR-10 with a 3x3 initial convolution and no max pooling to suit the dataset’s 32x32 images, achieving up to 93–95% validation accuracy with proper training.

<p align="center">
<img src="https://i.postimg.cc/mDT3D9Pr/image.png" width="800">
</p>

# Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
tqdm>=4.65.0
```

- See `requirements.txt` for a full list of dependencies.

# Project Structure

```bash
ResNet34/
├── data/
│   └── dataset.py
├── model/
│   ├── resnet.py
│   └── residual.py
├── checkpoints/
│   └── resnet34.pth
├── requirements.txt
├── stats.json
├── train.py
└── README.md
```

# Setup

Clone the repository:
```cmd
git clone https://github.com/buibaogianguyen/resnet34.git
cd resnet34
```

Install dependencies:
```cmd
pip install -r requirements.txt
```

If using a GPU, ensure the PyTorch version matches your CUDA toolkit (e.g., for CUDA 11.8, install `torch>=2.0.0+cu118`). Check [PyTorch's official site](https://pytorch.org/) for CUDA-specific installation.

Prepare the dataset:
The CIFAR-10 dataset is automatically downloaded by `dataset.py` to the `./data` directory during the first run of `train.py`. Ensure you have an internet connection, or verify that `./data/cifar-10-batches-py` exists.

# Usage

## Training

To train the model, run:
```cmd
python train.py
```

Configurations:
- **Epochs**: Default is 200 epochs (adjustable in `train.py`).
- **Batch Size**: Currently set to 256, adjust to 128 or 64 if limited memory.
- **Image Resolution**: Fixed at 32x32 for CIFAR-10, with data augmentation (RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing).
- **Hyperparameters**: Learning rate=0.1, weight decay=5e-4, MultiStepLR scheduler with milestones at epochs 60, 120, 160

The script saves:
- The best model checkpoint (`checkpoints/resnet34.pth`) based on validation accuracy.
- Validation metrics (`stats.json`) for accuracy.

# Performance

This shows the performance of the model over the first 70 epochs of training in terms of Validation Loss and Validation Accuracy over epochs. The model was trained to epoch 143 and reached a peak validation accuracy of 94.14%

<p align="left">
<img src="https://i.postimg.cc/SK7HXWx2/Graph-Val-Loss-Epoch.png" width="800">
</p>
<p align="right">
<img src="https://i.postimg.cc/Gmv5NkY7/Graph-Val-Acc-Epoch.png" width="800">
</p>


