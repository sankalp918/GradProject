This project focuses on image classification using Vision Transformers (ViT) for brain CT and MRI scans.

We have two databases: one containing CT scans and the other containing MRI scans. The CT scans are classified into two categories: **{0: "Normal"; 1: "Stroke"}**, while the MRI scans are classified into three categories: **{0: "Haemorrhagic"; 1: "Ischemic"; 2: "Normal"}**. Additionally, we have two datasets with CT scan images, which can be classified as balanced class datasets and imbalanced class datasets. Both datasets contain 120,000 CT images, with the primary difference being the class weightage ratio.

The objective of having these two separate datasets is to compare the performance of Vision Transformers (ViT) and VGG16 in deep learning image classification.

**After training and comparing all four models, it is evident that the ViT outperforms VGG16 in both cases. Furthermore, when comparing ViT and VGG16 models trained on balanced versus imbalanced datasets, it becomes clear that the models trained on the balanced dataset have a significant advantage over those trained on the imbalanced dataset.**

- ViT trained on an imbalanced class CT dataset:![image](https://github.com/user-attachments/assets/482587f7-5a28-4ad9-ae33-ebfba7500d4d)
- ViT trained on a balanced class CT dataset:![image](https://github.com/user-attachments/assets/b6c1bbe6-79c2-4181-a8c7-33c6381d3945)
- VGG16 trained on an imbalanced class CT dataset: **93% Accuracy**
- VGG16 trained on a balanced class CT dataset: **98.5% Accuracy**

Therefore, moving forward, there is no need to use VGG16 or an imbalanced class dataset for the MRI scan images.

- VGG16 trained on a balanced class MRI dataset:![image](https://github.com/user-attachments/assets/e57fcd00-df9b-4cea-b399-b5e65e63f57e)
