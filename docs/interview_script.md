# Interview Script

## Short Version

This project focused on underwater object detection for marine species, including holothurian, echinus, scallop, and starfish.

My main work was to build a complete detection pipeline. I converted XML annotations into YOLO format, prepared a fixed test set, and used 5-fold cross-validation for training and evaluation.

I compared several models, including YOLO11s, YOLO11m, RT-DETR-l, and Faster R-CNN. The models were evaluated using Precision, Recall, mAP50, and mAP50-95.

In addition to clean test evaluation, I also designed a robustness benchmark with blur, haze, and combined blur-haze degradation. This allowed me to analyze how model performance changed under degraded underwater image conditions.

The results showed that YOLO11s achieved the best overall performance on the clean test set. Under degraded conditions, performance decreased progressively, especially for combined blur and haze. The main failure cases were small scallops and low-contrast holothurians.

Through this project, I gained experience in dataset preprocessing, model training, evaluation, robustness testing, and failure case analysis.

## Key Points to Mention

- Complete object detection pipeline
- XML to YOLO annotation conversion
- Fixed test set and 5-fold cross-validation
- YOLO11, RT-DETR, and Faster R-CNN comparison
- Blur, haze, and combined degradation benchmark
- Precision, Recall, mAP50, and mAP50-95 evaluation
- Small-object and low-contrast failure cases

## Possible Interview Question

### What was the main challenge in this project?

The main challenge was that underwater images often contain low contrast, blur, haze, and camouflage-like backgrounds. Small scallops were difficult to detect because of their small size, while holothurians were difficult because they often had low contrast against the background.

### Why did you design a robustness benchmark?

Clean test performance does not fully show how reliable a model is under degraded underwater conditions. Therefore, I generated blur, haze, and combined degradation test sets to evaluate how model performance changes under different visual degradation conditions.

### What did you learn from the results?

I learned that YOLO11s performed best overall on the clean test set, but all models showed performance degradation under blur and haze. Combined blur and haze caused the strongest performance drop. I also learned that per-class failure analysis is important because different species have different detection challenges.