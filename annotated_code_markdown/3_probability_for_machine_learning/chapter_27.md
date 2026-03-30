# 概率论与机器学习
## Chapter 27

---

### Chapter Summary

# Chapter 27: ROC and Precision-Recall Curves

## Overview
This chapter explores two complementary evaluation frameworks for classification: **ROC curves** (sensitivity/specificity) and **Precision-Recall curves** (precision/recall). Each reveals different aspects of model performance.

## Key Concepts
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **Precision-Recall**: Positive predictive value vs sensitivity
- **AUC-ROC**: Area under ROC, invariant to classification threshold
- **F1 Score**: Harmonic mean of precision and recall
- **Use Cases**: Choose ROC or PR based on problem

## Evolution of Examples

### Two Complementary Frameworks
1. **01_roc_curve.py**: Plot ROC curve and calculate AUC-ROC
2. **02_precision_recall_f1.py**: Plot Precision-Recall curve and calculate F1

## ROC Curve (Receiver Operating Characteristic)

### Components
**True Positive Rate (Sensitivity/Recall)**
```
TPR = TP / (TP + FN)
```
Of actual positive cases, how many did we catch?

**False Positive Rate (1 - Specificity)**
```
FPR = FP / (FP + TN)
```
Of actual negative cases, how many did we incorrectly flag?

### How ROC Curve is Built
1. Sort samples by predicted probability
2. Threshold at top: All positive → TPR=1, FPR=1 (top-right)
3. Gradually lower threshold
4. Each threshold: Calculate TPR and FPR
5. At bottom: No positive predictions → TPR=0, FPR=0 (origin)
6. Plot all points → ROC curve

### Interpretation
- **Upper-left region (ideal)**: High TPR, low FPR
- **Diagonal line**: Random classifier (TPR=FPR)
- **Lower-right region (bad)**: Low TPR, high FPR

### AUC-ROC
```
AUC = Area under ROC curve
```
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC = 0.0: Inverted classifier (always wrong)

**Probabilistic interpretation**: AUC = P(classifier ranks random positive higher than random negative)

### When to Use ROC
- Balanced classes
- Cost of FP ≈ Cost of FN
- Want threshold-independent evaluation
- Comparing multiple classifiers

## Precision-Recall Curve

### Components
**Precision (Positive Predictive Value)**
```
Precision = TP / (TP + FP)
```
Of positive predictions, how many were correct?

**Recall (Sensitivity)**
```
Recall = TP / (TP + FN)
```
Of actual positive cases, how many did we catch?

### How PR Curve is Built
Similar to ROC:
1. Sort samples by predicted probability
2. Threshold at top: Precision=1, Recall=small (top-left)
3. Gradually lower threshold
4. Each threshold: Calculate precision and recall
5. At bottom: Precision=P(+), Recall=1 (bottom-right)
6. Plot → PR curve

### Interpretation
- **Upper-right region (ideal)**: High precision, high recall
- **Random classifier (imbalanced)**: Horizontal line at P(+)
- **Lower-left region (bad)**: Low precision, low recall

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall.
- F1 = 1.0: Perfect
- F1 = 0.0: Terrible
- Balanced: Emphasizes both metrics equally

### When to Use PR Curve
- **Imbalanced classes** (especially skewed to negative)
- **Cost of FP ≠ Cost of FN**
- Focus on positive class performance
- Rare event detection
- Fraud/anomaly detection

## ROC vs Precision-Recall: A Key Difference

### With Imbalanced Data (99% negative, 1% positive)

**ROC Curve**:
- Baseline (random): AUC = 0.5
- Even mediocre classifier: Might achieve AUC = 0.7
- FPR stays low (denominator is huge)
- Large negative class "masks" poor performance

**Precision-Recall Curve**:
- Baseline (random): AUC ≈ 0.01 (very low)
- Poor classifier: Low AUC
- Sensitive to positive class performance
- Clearly shows quality differences

### Example: Email Spam Detection
99% legitimate, 1% spam

Classifier A:
- Catches 90% of spam (TP=90)
- Flags 1% of legitimate as spam (FP=990)
- Precision = 90/(90+990) = 0.083 (terrible)
- Recall = 90/(90+10) = 0.9 (good)
- ROC AUC = high (good TPR, low FPR)
- PR AUC = low (terrible precision)

ROC masks the problem; PR reveals it clearly!

## Summary Comparison

| Aspect | ROC | Precision-Recall |
|--------|-----|------------------|
| Best for | Balanced classes | Imbalanced classes |
| Axes | TPR vs FPR | Precision vs Recall |
| Random baseline | Diagonal (0.5) | Horizontal at P(+) |
| Threshold shown | No | No |
| Interpretability | TPR/FPR tradeoff | Precision/Recall tradeoff |
| Sensitivity | Moderate to imbalance | High to imbalance |
| Common use | General classification | Anomaly/rare event |

## Practical Recommendation

### Always Report
1. **Balanced data**: ROC AUC + F1
2. **Imbalanced data**: PR AUC + F1
3. **Critical decisions**: Both ROC and PR curves

### Decision Framework
```
Is negative class underrepresented?
  Yes → Use Precision-Recall curve
  No → Use ROC curve

Are costs symmetric?
  Yes → F1 score is sufficient
  No → Plot curve, choose threshold manually
```

## Key Takeaways
1. ROC and PR curves are complementary: use both
2. ROC: Sensitivity vs specificity (good for balanced data)
3. PR: Precision vs recall (good for imbalanced data)
4. Imbalanced data: PR curve reveals truth better than ROC
5. AUC summarizes curve; F1 gives single threshold score
6. Always visualize curves, not just numbers
7. Choose metric based on problem structure and costs

---
