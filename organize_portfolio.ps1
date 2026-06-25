# ==============================
# Underwater Detection Portfolio Organizer
# ==============================

$SRC = "F:\Program Files\Underwater_Detection"
$DST = "F:\YYT_portfolio\underwater-object-detection-portfolio"

Write-Host "Source: $SRC"
Write-Host "Target: $DST"

# ---------- Create folder structure ----------

$folders = @(
    "configs",
    "data",
    "docs",
    "figures",
    "figures\paper_figures",
    "results",
    "scripts",
    "scripts\preprocessing",
    "scripts\training",
    "scripts\evaluation",
    "scripts\robustness",
    "scripts\augmentation"
)

foreach ($folder in $folders) {
    $path = Join-Path $DST $folder
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
        Write-Host "Created folder: $folder"
    }
}

# ---------- Fix .gitignore if it was created as a folder ----------

$gitignorePath = Join-Path $DST ".gitignore"

if (Test-Path $gitignorePath -PathType Container) {
    Remove-Item $gitignorePath -Recurse -Force
    Write-Host "Removed wrong .gitignore folder"
}

if (!(Test-Path $gitignorePath)) {
    New-Item -ItemType File -Force -Path $gitignorePath | Out-Null
    Write-Host "Created .gitignore file"
}

# ---------- Create basic files ----------

$basicFiles = @(
    "README.md",
    "requirements.txt",
    "data\README.md",
    "docs\project_summary.md",
    "docs\experiment_notes.md",
    "docs\interview_script.md",
    "results\baseline_clean_results.csv",
    "results\robustness_results.csv",
    "results\per_class_results.csv"
)

foreach ($file in $basicFiles) {
    $path = Join-Path $DST $file
    if (!(Test-Path $path)) {
        New-Item -ItemType File -Force -Path $path | Out-Null
        Write-Host "Created file: $file"
    }
}

# ---------- Helper function ----------

function Copy-IfExists {
    param (
        [string]$FileName,
        [string]$TargetSubFolder
    )

    $sourcePath = Join-Path $SRC $FileName
    $targetPath = Join-Path (Join-Path $DST $TargetSubFolder) $FileName

    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $targetPath -Force
        Write-Host "Copied: $FileName -> $TargetSubFolder"
    } else {
        Write-Host "Missing: $FileName" -ForegroundColor Yellow
    }
}

# ---------- Copy configs ----------

$configFiles = @(
    "data.yaml",
    "original_test.yaml",
    "test_only.yaml"
)

foreach ($file in $configFiles) {
    Copy-IfExists $file "configs"
}

# ---------- Copy preprocessing scripts ----------

$preprocessingFiles = @(
    "convert_data.py",
    "re-split_test.py",
    "duplicate_images_check.py",
    "check_dataset_count.py",
    "check_label_quality.py"
)

foreach ($file in $preprocessingFiles) {
    Copy-IfExists $file "scripts\preprocessing"
}

# ---------- Copy training scripts ----------

$trainingFiles = @(
    "train.py",
    "kfold_train.py",
    "kfold_train_yolo11m_640.py",
    "kfold_train_yolo11n_640.py",
    "kfold_train_aug.py",
    "kfold_train_clsaug_768.py",
    "rt-detr_train.py",
    "faster_rcnn_train.py",
    "faster_rcnn_train_fast.py"
)

foreach ($file in $trainingFiles) {
    Copy-IfExists $file "scripts\training"
}

# ---------- Copy augmentation scripts ----------

$augmentationFiles = @(
    "generate_augmented_dataset.py",
    "generate_clsaug_dataset.py"
)

foreach ($file in $augmentationFiles) {
    Copy-IfExists $file "scripts\augmentation"
}

# ---------- Copy robustness scripts ----------

$robustnessFiles = @(
    "generate_robustness_benchmark.py",
    "generate_robustness_benchmark_2.0.py",
    "eval_robustness_benchmark_5fold.py",
    "eval_faster_rcnn_robustness.py",
    "make_combined_robustness_figure.py"
)

foreach ($file in $robustnessFiles) {
    Copy-IfExists $file "scripts\robustness"
}

# ---------- Copy evaluation scripts ----------

$evaluationFiles = @(
    "collect_rtdetr_test_results.py",
    "collect_test_results.py",
    "collect_yolo11s_768_test_results.py",
    "collect_yolo11s_768_val_results.py",
    "collect_yolo11s_aug_test_results.py",
    "collect_yolo11s_aug_val_results.py",
    "collect_yolo11s_test_results.py",
    "collect_yolo11s_val_results.py",
    "compare_val.py",
    "val_test.py",
    "test_model.py",
    "find_holothurian_missed_cases.py",
    "generate_failure_case_predictions.py",
    "make_failure_case_montage.py",
    "reveal_faster_rcnn_corrected.py",
    "visual_check_holothurian_scallop.py",
    "check_aug_fold1_class_metrics.py",
    "check_clsaug_fold1_class_metrics.py",
    "check_original_640_fold1_class_metrics.py",
    "check_original_768_class_metrics.py"
)

foreach ($file in $evaluationFiles) {
    Copy-IfExists $file "scripts\evaluation"
}

# ---------- Copy result CSV files ----------

$resultFiles = @(
    "clsaug_fold1_class_metrics.csv"
)

foreach ($file in $resultFiles) {
    Copy-IfExists $file "results"
}

# ---------- Copy figures ----------

$figureFiles = @(
    "robustness_blur_map50.png",
    "robustness_curves_combined.pdf",
    "robustness_curves_combined.png",
    "robustness_haze_map50.pdf",
    "robustness_haze_map50.png"
)

foreach ($file in $figureFiles) {
    Copy-IfExists $file "figures"
}

# ---------- Copy paper_figures folder ----------

$paperFiguresSrc = Join-Path $SRC "paper_figures"
$paperFiguresDst = Join-Path $DST "figures\paper_figures"

if (Test-Path $paperFiguresSrc) {
    Copy-Item "$paperFiguresSrc\*" $paperFiguresDst -Recurse -Force
    Write-Host "Copied folder: paper_figures -> figures\paper_figures"
} else {
    Write-Host "Missing folder: paper_figures" -ForegroundColor Yellow
}

# ---------- Write .gitignore content ----------

$gitignoreContent = @"
runs/
weights/
*.pt
*.pth
*.onnx
data/images/
data/labels/
yolo_dataset/
yolo_dataset_alsaug/
yolo_dataset_augmented/
yolo_dataset_clsaug/
__pycache__/
.cache/
*.log
.DS_Store
_archive_old/
"@

Set-Content -Path $gitignorePath -Value $gitignoreContent -Encoding UTF8
Write-Host "Updated .gitignore"

Write-Host ""
Write-Host "Done. Portfolio files have been organized." -ForegroundColor Green