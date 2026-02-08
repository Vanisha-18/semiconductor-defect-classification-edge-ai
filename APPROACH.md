# Technical Approach - IESA DeepTech Hackathon 2026

## Problem Analysis
Edge AI-based defect classification for semiconductor manufacturing with constraints:
- Model size: <15 MB
- Inference time: <100 ms
- Deployment: NXP eIQ platform

## Dataset Preparation
### Initial Challenge
- Collected 27,900 raw semiconductor microscopy images
- No pre-labeled data available

### Solution: CLIP-based Zero-Shot Classification
- Used OpenAI CLIP (ViT-base-patch32) for automated labeling
- Defined 8 semantic categories with descriptive prompts
- Implemented confidence-based filtering (threshold: 0.70 for Clean class)
- Achieved automated organization into Train/Val/Test (80/10/10)

### Dataset Statistics
- Total Images: 27,900
- Classes: 8 (Bridges, Opens, LER, Malformed_Vias, CMP_Scratches, Cracks, Clean, Others)
- Severe imbalance: LER (67%), Clean (<1%)

## Model Architecture Selection

### Requirements Analysis
| Constraint | Target | Solution |
|------------|--------|----------|
| Model Size | <15 MB | MobileNetV2 (lightweight) |
| Inference Speed | <100 ms | Efficient architecture |
| Edge Deployment | NXP eIQ | ONNX format |
| Accuracy | Maximize | Transfer learning |

### Final Architecture
```
Input (224x224x3)
    ↓
MobileNetV2 Base (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (8, Softmax)

Total Parameters: 2,422,984
Trainable: 2,386,184
Model Size: 9.11 MB (ONNX)
```

## Training Strategy

### Phase 1: Frozen Base (10 epochs)
- Objective: Train classification head only
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-entropy
- Class Weights: Balanced (to handle imbalance)
- Data Augmentation:
  - Rotation: ±20°
  - Shifts: ±20%
  - Flips: Horizontal & Vertical
  - Zoom: ±20%

### Phase 2: Fine-tuning (10 epochs)
- Objective: End-to-end optimization
- Optimizer: Adam (lr=0.0001)
- Unfroze all layers
- Continued with class weights

### Results
- Validation Accuracy (Peak): 71.40%
- Test Accuracy (Final): 69.93%
- Model Size: 9.11 MB
- Inference Time: 90-110 ms

## Challenges & Solutions

### Challenge 1: Extreme Class Imbalance
**Problem**: LER class dominated 67% of dataset
**Solution**: 
- Computed balanced class weights
- Applied weighted loss function
- Result: Prevented model from only predicting LER

### Challenge 2: Rare Classes (Clean, Malformed_Vias)
**Problem**: <1% representation
**Solution**:
- Aggressive data augmentation
- Monitored per-class metrics
- Result: Maintained some predictive capability

### Challenge 3: Compute Resource Limitations
**Problem**: Free Colab GPU timeout
**Solution**:
- Implemented auto-save checkpoints
- 2-phase training (10+10 epochs)
- Result: Completed training before timeout

## Edge Deployment Readiness

### ONNX Conversion
```python
tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)],
    opset=13
)
```

### Deployment Specs
- Format: ONNX
- Size: 9.11 MB (60% under limit)
- Quantization-ready: INT8 capable
- Platform: NXP eIQ compatible

## Future Improvements
1. **Data Collection**: Gather more balanced dataset
2. **Architecture Search**: Test EfficientNet-B0
3. **Ensemble**: Combine multiple models
4. **Active Learning**: Focus on misclassified samples
5. **Quantization**: INT8 optimization for faster inference
