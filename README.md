# Liver-Seg

# Liver Segmentation from CT Images

## Overview
This project focuses on liver segmentation from CT images using a deep learning-based U-Net model. The goal is to accurately segment the liver region from medical imaging scans.

## Dataset
- **Source**: [3d_liver-segmentation](https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation)
- **Modality**: CT Scans
- **Annotations**: Liver masks for segmentation

## Methodology
1. **Preprocessing**
   - Resizing images for compatibility with the U-Net model
   - Normalization of pixel values
2. **Model Architecture**
   - U-Net (self-defined architecture)
3. **Training**
   - Loss Function: Binary Cross-Entropy (BCE)
   - Optimizer: Adam
   - Metrics:
          - Dice Coefficient
          - mIoU
          - Precision
          - Recall
4. **Inference**
   - Model tested on unseen images
   - Segmented liver regions visualized


- **Visualization**
  - Example segmented outputs compared with ground truth masks
![image](https://github.com/user-attachments/assets/feada266-e08d-407d-be8f-5d98718d63c4)
## Future Improvements
- Experiment with different loss functions such as Dice Loss
- Implement post-processing techniques to refine segmentation
- Explore ensembling with other architectures (e.g., DeepLabV3+)

## Contributors
- [Shankar](https://github.com/ShankarOmmi)
- [Thanuja](https://github.com/thanuja-bobbepalli)

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

