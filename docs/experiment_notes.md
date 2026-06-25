# Experiment Notes

## Dataset

- Total images: 5455
- Fixed test set: 550 images
- Training and validation set: 4905 images
- Cross-validation: 5-fold
- Classes:
  - holothurian
  - echinus
  - scallop
  - starfish

## Annotation Conversion

Original annotations were stored in XML format. They were converted into YOLO format, where each bounding box is represented by normalized center coordinates, width, and height.

## Models

The following models were evaluated:

- YOLO11s
- YOLO11m
- RT-DETR-l
- Faster R-CNN ResNet50-FPN

## Evaluation Metrics

The models were evaluated using:

- Precision
- Recall
- mAP50
- mAP50-95

## Clean Test Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11s | 0.8343 | 0.7946 | 0.8590 | 0.5112 |
| YOLO11m | 0.8299 | 0.8004 | 0.8532 | 0.5091 |
| RT-DETR-l | 0.8050 | 0.7882 | 0.8478 | 0.4891 |
| Faster R-CNN | 0.8203 | 0.7516 | 0.8409 | 0.4711 |

## Robustness Benchmark

Three types of degradation were generated:

- Blur
- Haze
- Combined blur and haze

Each degradation type contains low, medium, and high severity levels.

## Observations

- Blur caused progressive performance degradation as severity increased.
- Haze had a stronger effect on low-contrast targets.
- Combined blur and haze caused the largest performance drop.
- Holothurian and scallop were the most challenging classes.
- Echinus and starfish were relatively more stable.

## Limitations

- Faster R-CNN was evaluated only on fold 1.
- The degradation benchmark was synthetically generated.
- Real underwater degradation may be more complex than the simulated settings.