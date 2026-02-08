# Results & Analysis

## Overall Performance
- **Test Accuracy**: 69.93%
- **Model Size**: 9.11 MB
- **Inference Time**: ~95 ms (average)

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bridges | 0.06 | 0.50 | 0.11 | 2 |
| CMP_Scratches | 0.31 | 0.50 | 0.38 | 413 |
| Clean | 0.02 | 0.33 | 0.04 | 3 |
| Cracks | 0.03 | 0.24 | 0.05 | 17 |
| LER | 0.93 | 0.74 | 0.82 | 2343 |
| Malformed_Vias | 0.00 | 0.00 | 0.00 | 1 |
| Opens | 0.07 | 0.50 | 0.13 | 6 |
| Others | 0.13 | 0.40 | 0.20 | 5 |

## Key Insights
- **Strong Performance**: LER class (93% precision, 74% recall)
- **Challenges**: Rare classes struggle due to limited data
- **Balanced Approach**: Weighted loss prevented catastrophic overfitting

## Edge Deployment Metrics
✅ Size: 9.11 MB < 15 MB (passed)
✅ Inference: ~95 ms < 100 ms (passed)
✅ Format: ONNX (compatible)
