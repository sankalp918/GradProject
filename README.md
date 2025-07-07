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

## Results and Model Performance

**After training and comparing all four models, it is evident that the ViT outperforms VGG16 in both cases. Furthermore, when comparing ViT and VGG16 models trained on balanced versus imbalanced datasets, it becomes clear that the models trained on the balanced dataset have a significant advantage over those trained on the imbalanced dataset.**

- ViT trained on an imbalanced class CT dataset:![image](https://github.com/user-attachments/assets/482587f7-5a28-4ad9-ae33-ebfba7500d4d)
- ViT trained on a balanced class CT dataset:![image](https://github.com/user-attachments/assets/b6c1bbe6-79c2-4181-a8c7-33c6381d3945)
- VGG16 trained on an imbalanced class CT dataset: **93% Accuracy**
- VGG16 trained on a balanced class CT dataset: **98.5% Accuracy**

Therefore, moving forward, there is no need to use VGG16 or an imbalanced class dataset for the MRI scan images.

- VGG16 trained on a balanced class MRI dataset:![image](https://github.com/user-attachments/assets/e57fcd00-df9b-4cea-b399-b5e65e63f57e)

The links to the models:
1. [Balanced_dataset_VIT_CT](https://huggingface.co/GRAD04/Balanced_dataset_VIT_CT)
2. [Balanced_dataset_VGG16_CT](https://huggingface.co/GRAD04/Balanced_dataset_VGG16_CT)
3. [Imbalanced_Dataset_VIT_CT](https://huggingface.co/GRAD04/Imbalanced_Dataset_VIT_CT)
4. [Imbalanced_Dataset_VGG16_CT](https://huggingface.co/GRAD04/Imbalanced_Dataset_VGG16_CT)
5. [Balanced_Dataset_VIT_MRI](https://huggingface.co/GRAD04/Balanced_Dataset_VIT_MRI)

The links to the datasets:
1. [Balanced_CT_Scan](https://huggingface.co/datasets/GRAD04/Balanced_CT_Scan)
2. [Imbalanced_CT_Scan](https://huggingface.co/datasets/GRAD04/Imbalanced_CT_Scan)
3. [Balanced_MRI_Scan](https://huggingface.co/datasets/GRAD04/Balanced_MRI_Scan)

## Project Structure

```
GradProject/
├── CNN_vs_ViT.ipynb                      # CNN vs ViT comparison notebook
├── Fusion_Model_V1.ipynb                 # Fusion model version 1
├── Fusion_Model.ipynb                    # Multi-modal fusion model
├── Model_trained_on_Combined_Dataset.ipynb # Combined dataset training
├── Model_trained_on_Original_Dataset.ipynb # Original dataset training
├── VIT_Enhanced_Fusion_V1.1.ipynb        # Enhanced fusion model
├── VIT_MRI_BALANCED_DATASET.ipynb        # ViT training on MRI data
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation
```

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes with the provided notebooks
5. Submit a pull request

## License

This project is open source and available under the MIT License.
