# Project Summary

## Project Title

Underwater Object Detection and Robustness Evaluation

## Background

Underwater object detection is challenging because underwater images often suffer from low contrast, motion blur, haze, and background camouflage. In this project, the goal was to detect four types of marine species: holothurian, echinus, scallop, and starfish.

## Objective

The objective of this project was to build a complete object detection pipeline and evaluate model performance under both clean and degraded underwater image conditions.

## My Contributions

- Converted XML annotations into YOLO format.
- Prepared a fixed test set and 5-fold train/validation splits.
- Trained and evaluated YOLO11, RT-DETR, and Faster R-CNN models.
- Designed synthetic robustness benchmarks using blur, haze, and combined blur-haze degradation.
- Collected and compared model performance using Precision, Recall, mAP50, and mAP50-95.
- Analyzed failure cases for small and low-contrast underwater objects.

## Methods

The dataset was first converted from XML annotation format to YOLO format. A fixed test set of 550 images was used for final evaluation, while the remaining 4905 images were used for 5-fold cross-validation.

Several object detection models were evaluated, including YOLO11s, YOLO11m, RT-DETR-l, and Faster R-CNN with ResNet50-FPN.

To test robustness, synthetic degraded test sets were generated using motion blur, haze, and combined blur-haze degradation at different severity levels.

## Results

YOLO11s achieved the best overall performance on the clean test set, with a mAP50 of 0.8590 and mAP50-95 of 0.5112.

The robustness experiments showed that model performance decreased under degraded image conditions. Combined blur and haze caused the largest performance drop.

## Failure Cases

The main failure cases were:

- Small scallops, which were often missed due to small object size.
- Low-contrast holothurians, which were difficult to distinguish from the underwater background.

## What I Learned

Through this project, I gained practical experience in dataset preprocessing, object detection model training, cross-validation, robustness evaluation, result collection, and failure case analysis.

This project also helped me understand how model performance can change under realistic image degradation conditions.