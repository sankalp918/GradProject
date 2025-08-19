# Brain CT and MRI Classification using Vision Transformers

## Project Overview

This project focuses on image classification using Vision Transformers (ViT) for brain CT and MRI scans.

We have two databases: one containing CT scans and the other containing MRI scans. The CT scans are classified into two categories: **{0: "Normal"; 1: "Stroke"}**, while the MRI scans are classified into three categories: **{0: "Haemorrhagic"; 1: "Ischemic"; 2: "Normal"}**. Additionally, we have two datasets with CT scan images, which can be classified as balanced class datasets and imbalanced class datasets. Both datasets contain 120,000 CT images, with the primary difference being the class weightage ratio.

The objective of having these two separate datasets is to compare the performance of Vision Transformers (ViT) and VGG16 in deep learning image classification.

## Installation and Setup

### Requirements

This project requires Python 3.8+ and the following dependencies:

```bash
pip install -r requirements.txt
```

The requirements.txt file includes:
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Computer Vision**: torchvision, opencv-python, Pillow
- **Machine Learning**: transformers, scikit-learn
- **Data Analysis**: numpy, pandas
- **Visualization**: matplotlib, seaborn

### GPU Support (Recommended)

For CUDA-enabled GPU support with PyTorch:

```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open any of the Jupyter notebooks to explore the models:
   - `CNN_vs_ViT.ipynb` - Comparison between CNN and ViT models
   - `VIT_MRI_BALANCED_DATASET.ipynb` - ViT training on MRI data
   - `Fusion_Model.ipynb` - Multi-modal fusion approaches
   - `Model_trained_on_Combined_Dataset.ipynb` - Combined dataset training

## Dataset Setup

All datasets are hosted on Hugging Face and can be downloaded automatically within the notebooks. The following datasets are used:

### CT Scan Datasets
- **Balanced CT Dataset**: 120,000 images with equal class distribution
  - Repository: [GRAD04/Balanced_CT_Scan](https://huggingface.co/datasets/GRAD04/Balanced_CT_Scan)
  - Classes: Normal, Stroke (binary classification)
  
- **Imbalanced CT Dataset**: 120,000 images with imbalanced class distribution
  - Repository: [GRAD04/Imbalanced_CT_Scan](https://huggingface.co/datasets/GRAD04/Imbalanced_CT_Scan)
  - Classes: Normal, Stroke (binary classification)

### MRI Scan Dataset
- **Balanced MRI Dataset**: 108,000 images with equal class distribution
  - Repository: [GRAD04/Balanced_MRI_Scan](https://huggingface.co/datasets/GRAD04/Balanced_MRI_Scan)
  - Classes: Haemorrhagic, Ischemic, Normal (3-class classification)

### Automatic Download in Notebooks

The notebooks include code to automatically download datasets from Hugging Face:

```python
from huggingface_hub import snapshot_download

# Download CT dataset
ct_path = snapshot_download(
    repo_id="GRAD04/Balanced_CT_Scan",
    repo_type="dataset",
    local_dir="./data/"
)

# Download MRI dataset  
mri_path = snapshot_download(
    repo_id="GRAD04/Balanced_MRI_Scan",
    repo_type="dataset", 
    local_dir="./data/"
)
```

## Pre-trained Models

All trained models are available on Hugging Face Hub:

### CT Scan Models
| Model | Dataset | Accuracy | Repository |
|-------|---------|----------|------------|
| **ViT** | Balanced | **99.2%** | [Balanced_dataset_VIT_CT](https://huggingface.co/GRAD04/Balanced_dataset_VIT_CT) |
| **VGG16** | Balanced | **98.5%** | [Balanced_dataset_VGG16_CT](https://huggingface.co/GRAD04/Balanced_dataset_VGG16_CT) |
| **ViT** | Imbalanced | **97.8%** | [Imbalanced_Dataset_VIT_CT](https://huggingface.co/GRAD04/Imbalanced_Dataset_VIT_CT) |
| **VGG16** | Imbalanced | **93.0%** | [Imbalanced_Dataset_VGG16_CT](https://huggingface.co/GRAD04/Imbalanced_Dataset_VGG16_CT) |

### MRI Scan Models
| Model | Dataset | Accuracy  | Repository |
|-------|---------|-----------|------------|
| **ViT** | Balanced | **99.92%** | [Balanced_Dataset_VIT_MRI](https://huggingface.co/GRAD04/Balanced_Dataset_VIT_MRI) |
| **VGG16** | Balanced | **97.0%** | [Balanced_Dataset_VGG16_MRI](https://huggingface.co/GRAD04/Balanced_MRI_VGG16) |

### Fusion Models
The project also includes advanced fusion models that combine CT and MRI modalities:
- **Dual-ViT Fusion Model**: Combines pretrained ViT models for both CT and MRI
- **Enhanced Fusion Model**: Advanced attention-based fusion with cross-modal learning
- **Model artifacts**: Generated locally during training (saved as `.pth` files)

## Using Pre-trained Models

### Download Models from Hugging Face

```python
from huggingface_hub import snapshot_download

# Download CT ViT model
model_path = snapshot_download(
    repo_id="GRAD04/Balanced_dataset_VIT_CT",
    local_dir="./models/"
)

# Download MRI ViT model
model_path = snapshot_download(
    repo_id="GRAD04/Balanced_Dataset_VIT_MRI", 
    local_dir="./models/"
)
```

### Model Loading Example

```python
from transformers import ViTForImageClassification
import torch

# Load CT model
ct_model = ViTForImageClassification.from_pretrained(
    "GRAD04/Balanced_dataset_VIT_CT"
)

# Load MRI model  
mri_model = ViTForImageClassification.from_pretrained(
    "GRAD04/Balanced_Dataset_VIT_MRI"
)
```

## Results and Model Performance

### Performance Summary

**After training and comparing all models, it is evident that the ViT outperforms VGG16 in both cases. Furthermore, when comparing ViT and VGG16 models trained on balanced versus imbalanced datasets, it becomes clear that the models trained on the balanced dataset have a significant advantage over those trained on the imbalanced dataset.**

### Performance Visualizations

- ViT trained on an imbalanced class CT dataset:![image](https://github.com/user-attachments/assets/482587f7-5a28-4ad9-ae33-ebfba7500d4d)
- VIT trained on a balanced class CT dataset:![image](https://github.com/user-attachments/assets/b6c1bbe6-79c2-4181-a8c7-33c6381d3945)
- VIT trained on a balanced class MRI dataset:![image](https://github.com/user-attachments/assets/e57fcd00-df9b-4cea-b399-b5e65e63f57e)

### Key Findings
- **ViT consistently outperforms VGG16** across all datasets
- **Balanced datasets** provide significant performance improvements over imbalanced datasets  
- **MRI ViT model** achieves exceptional performance (99.92% accuracy)
- **Fusion models** demonstrate the potential for multi-modal medical image analysis
- Therefore, moving forward, there is no need to use VGG16 or an imbalanced class dataset for the MRI scan images.

## Project Structure

```
GradProject/
├── CNN_vs_ViT.ipynb                      # CNN vs ViT comparison notebook
├── Fusion_Model_V1.ipynb                 # Fusion model version 1
├── Fusion_Model.ipynb                    # Multi-modal fusion model
├── Model_trained_on_Combined_Dataset.ipynb # Combined dataset training
├── Model_trained_on_Original_Dataset.ipynb # Original dataset training
├── VIT_Enhanced_Fusion_V1.1.ipynb        # Enhanced fusion model with attention
├── VIT_MRI_BALANCED_DATASET.ipynb        # ViT training on MRI data
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation
```

### Notebook Descriptions

- **`CNN_vs_ViT.ipynb`**: Comprehensive comparison between CNN (VGG16) and Vision Transformer models
- **`VIT_MRI_BALANCED_DATASET.ipynb`**: ViT model training specifically for MRI classification (3 classes)
- **`Model_trained_on_Combined_Dataset.ipynb`**: CNN model training on combined CT datasets
- **`Model_trained_on_Original_Dataset.ipynb`**: Training on original dataset configurations
- **`Fusion_Model.ipynb`**: Basic multi-modal fusion combining CT and MRI modalities
- **`Fusion_Model_V1.ipynb`**: First version of fusion model implementation
- **`VIT_Enhanced_Fusion_V1.1.ipynb`**: Advanced fusion model with cross-attention mechanisms

## Reproducing Results

### Training from Scratch

1. **Install dependencies**: Follow the installation instructions above
2. **Download datasets**: Use the Hugging Face Hub integration in notebooks
3. **Run individual notebooks**: Each notebook is self-contained and includes data loading
4. **GPU recommended**: Training requires CUDA-compatible GPU for reasonable training times

### Expected Training Times (A100 GPU)
- **Single ViT model**: ~2-3 hours for 25 epochs
- **Fusion models**: ~4-6 hours depending on complexity
- **VGG16 models**: ~1-2 hours for 15 epochs

### Hardware Requirements
- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 24GB+ GPU memory (A100, RTX 3090, etc.), 32GB+ RAM
- **Storage**: ~15GB for all datasets and model artifacts

## Technical Details

### Model Architectures
- **ViT Base**: `google/vit-base-patch16-224` as backbone
- **Input Size**: 224×224 RGB images
- **CT Classification**: Binary (Normal/Stroke)
- **MRI Classification**: Multi-class (Haemorrhagic/Ischemic/Normal)
- **Fusion**: Cross-attention between ViT embeddings from both modalities

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with scheduler
- **Batch Size**: 128-256 depending on available memory
- **Data Augmentation**: Standard vision augmentations applied
- **Early Stopping**: Implemented to prevent overfitting

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes with the provided notebooks
5. Submit a pull request

## License

This project is open source and available under the MIT License.