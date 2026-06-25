# Dataset Description

The original dataset is not included in this repository due to storage size and usage restrictions.

## Classes

The dataset contains four underwater marine species:

| Class ID | Class       |
| -------- | ----------- |
| 0        | holothurian |
| 1        | echinus     |
| 2        | scallop     |
| 3        | starfish    |

## Dataset Split

The dataset contains 5455 images in total.

* Fixed test set: 550 images
* Training and validation set: 4905 images
* 5-fold cross-validation:

  * Training images per fold: 3924
  * Validation images per fold: 981

## Annotation Format

The original annotations were stored in XML format and converted into YOLO format.

Each YOLO annotation line contains:

```text
class_id center_x center_y width height
```

All bounding box coordinates are normalized by image width and height.
