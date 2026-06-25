from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 保存路径
# ============================================================
PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")
OUTPUT_DIR = PROJECT_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据
# ============================================================
levels = ["Clean", "Low", "Medium", "High"]
x = np.arange(len(levels))

# -------------------------
# Blur
# -------------------------
blur_yolo11s = [0.8590, 0.7602, 0.5902, 0.4675]
blur_yolo11m = [0.8581, 0.7607, 0.5833, 0.4587]
blur_rtdetr  = [0.8478, 0.7396, 0.5637, 0.4318]

blur_yolo11s_std = [0.0, 0.0070, 0.0057, 0.0068]
blur_yolo11m_std = [0.0, 0.0044, 0.0059, 0.0041]
blur_rtdetr_std  = [0.0, 0.0082, 0.0109, 0.0151]

# -------------------------
# Haze
# -------------------------
haze_yolo11s = [0.8590, 0.8442, 0.7687, 0.5570]
haze_yolo11m = [0.8581, 0.8474, 0.7883, 0.6230]
haze_rtdetr  = [0.8478, 0.8396, 0.8107, 0.7242]

haze_yolo11s_std = [0.0, 0.0015, 0.0039, 0.0092]
haze_yolo11m_std = [0.0, 0.0056, 0.0063, 0.0193]
haze_rtdetr_std  = [0.0, 0.0026, 0.0090, 0.0190]

# -------------------------
# Combined Blur + Haze
# -------------------------
comb_yolo11s = [0.8590, 0.7434, 0.4968, 0.2202]
comb_yolo11m = [0.8581, 0.7415, 0.4956, 0.2296]
comb_rtdetr  = [0.8478, 0.7274, 0.5017, 0.2703]

comb_yolo11s_std = [0.0, 0.0042, 0.0045, 0.0118]
comb_yolo11m_std = [0.0, 0.0063, 0.0117, 0.0195]
comb_rtdetr_std  = [0.0, 0.0048, 0.0114, 0.0145]

# ============================================================
# 作图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

def plot_one(ax, title, y1, y2, y3, s1, s2, s3, sublabel):
    ax.errorbar(x, y1, yerr=s1, marker='o', linewidth=2, capsize=4, label='YOLO11s')
    ax.errorbar(x, y2, yerr=s2, marker='o', linewidth=2, capsize=4, label='YOLO11m')
    ax.errorbar(x, y3, yerr=s3, marker='o', linewidth=2, capsize=4, label='RT-DETR-l')

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0.15, 0.92)
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("mAP50")
    ax.set_title(f"({sublabel}) {title}", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

# 三个子图
plot_one(
    axes[0],
    "Motion Blur",
    blur_yolo11s, blur_yolo11m, blur_rtdetr,
    blur_yolo11s_std, blur_yolo11m_std, blur_rtdetr_std,
    "a"
)

plot_one(
    axes[1],
    "Haze",
    haze_yolo11s, haze_yolo11m, haze_rtdetr,
    haze_yolo11s_std, haze_yolo11m_std, haze_rtdetr_std,
    "b"
)

plot_one(
    axes[2],
    "Blur + Haze",
    comb_yolo11s, comb_yolo11m, comb_rtdetr,
    comb_yolo11s_std, comb_yolo11m_std, comb_rtdetr_std,
    "c"
)

# 只保留一个总图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存
png_path = OUTPUT_DIR / "robustness_curves_combined.png"
pdf_path = OUTPUT_DIR / "robustness_curves_combined.pdf"

plt.savefig(png_path, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print("Saved:")
print(png_path)
print(pdf_path)