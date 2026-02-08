# semiconductor-defect-classification-edge-ai
Edge-AI Semiconductor Defect Classification - IESA DeepTech Hackathon Phase 1
# IESA DeepTech Hackathon 2026 - Edge AI Defect Classification

## Team Information
- **Team Name**:PIXELS
- **MemberS**: Y Navadeep Saran, N Vanisha, Ashwini Tatikole, Ravi Tanuja

## Problem Statement
Edge AI-based defect classification system for semiconductor wafer/die images

## Results
- **Test Accuracy**: 69.93%
- **Model Size**: 9.11 MB (ONNX format)
- **Architecture**: MobileNetV2 with custom classification head

## Dataset
- **Total Images**: 27,900
- **Classes**: 8 (Bridges, Opens, LER, Malformed_Vias, CMP_Scratches, Cracks, Clean, Others)
- **Split**: 80% Train / 10% Validation / 10% Test
- **Preprocessing**: CLIP-based automated organization

## Model Architecture
- **Base**: MobileNetV2 (ImageNet pre-trained)
- **Input**: 224×224×3 RGB images
- **Custom Head**:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense(128, ReLU)
  - Dropout (0.3)
  - Dense(8, Softmax)
- **Total Parameters**: 2,422,984

## Training Strategy
### Phase 1: Frozen Base (10 epochs)
- Optimizer: Adam (lr=0.001)
- Trained classification head only

### Phase 2: Fine-tuning (10 epochs)
- Optimizer: Adam (lr=0.0001)
- Unfroze all layers for end-to-end training

### Code Structure:
- Codebase is modularized for clarity: Use train.py to replicate our results and inference.py to deploy the model on your local environment.

## Key Features
- **Class Imbalance Handling**: Implemented balanced class weights
- **Data Augmentation**: Rotation, shifts, flips, zoom
- **Edge-Ready**: 9.11 MB model suitable for NXP eIQ deployment
- **Fast Inference**: <100ms target on edge devices

## Challenges & Solutions
### Challenge 1: Severe Class Imbalance
- **Issue**: LER class dominated 67% of dataset
- **Solution**: Computed balanced class weights for loss function

### Challenge 2: Limited Rare Class Samples
- **Issue**: Clean (24), Cracks (135), Others (47) samples
- **Solution**: Heavy data augmentation + weighted sampling

## Performance Breakdown
```
Class              Precision  Recall  F1-Score  Support
Bridges            0.06       0.50    0.11      2
CMP_Scratches      0.31       0.50    0.38      413
Clean              0.02       0.33    0.04      3
Cracks             0.03       0.24    0.05      17
LER                0.93       0.74    0.82      2343
Malformed_Vias     0.00       0.00    0.00      1
Opens              0.07       0.50    0.13      6
Others             0.13       0.40    0.20      5
```

**Overall Accuracy**: 69.93%

## 📺 Demo Video
To see the semiconductor defect detection system in action, watch our demonstration video:
[Watch the Demonstration Video Here](./demo/iesa_demo.mp4) 
[View Dataset on Google Drive](https://drive.google.com/file/d/1qByS3AvmUSoQKtkTevO7nTg5aLYHIlcC/view?usp=sharing)

## Repository Structure
```
├── iesa_model.onnx          # Trained ONNX model
├── README.md                 # This file
└── requirements.txt          # Dependencies
```

## Requirements
```
tensorflow>=2.13.0
tf2onnx>=1.15.0
onnx>=1.14.0
numpy>=1.24.0
pillow>=10.0.0
scikit-learn>=1.3.0
```

## Deployment
Model is compatible with NXP eIQ platform for i.MX RT series edge devices.

## Future Improvements
1. Collect more balanced training data for rare classes
2. Implement ensemble methods for better generalization
3. Test real-time inference on actual edge hardware
4. Explore architecture search for better accuracy/size tradeoff
```

**requirements.txt**:
```
tensorflow>=2.13.0
tf2onnx>=1.15.0
onnx>=1.14.0
numpy>=1.24.0
pillow>=10.0.0
scikit-learn>=1.3.0
