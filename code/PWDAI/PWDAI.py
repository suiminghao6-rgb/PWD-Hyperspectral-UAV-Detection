# pwadai_from_paper.py
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------- USER CONFIG (改这三项路径即可) ----------------
sample_csv = r"E:\Alexnet band data_Healthy & Disease.csv"
image_tif   = r"E:\Original hyperspectral image.tif"
output_tif  = r""
# -----------------------------------------------------------------

# Expected number of spectral bands in sample CSV (excluding 'type')
n_bands_expected = 240

# PWDAI uses these 1-based band numbers (per the formula you gave)
B61 = 61
B112 = 112
B141 = 141

# block size (rows) for memory-safe processing
block_rows = 256

# nodata value to write in output (uint8)
nodata_value = 255

# ---------------- formula (strictly as provided) ----------------
def compute_pwadai(B61_arr, B112_arr, B141_arr):
    const_term = (678.056 - 548.031) * (751.963 - 548.031)
    data_term = (B112_arr - B61_arr) * (B141_arr - B61_arr)
    return np.abs(const_term - data_term) / 2.0

# ---------------- 1) read sample csv and compute sample PWDAI -------------
print("Reading sample CSV:", sample_csv)
df = pd.read_csv(sample_csv)

if 'type' not in df.columns:
    raise RuntimeError("样点 CSV 必须包含 'type' 列（0/1/2/3）。请确认文件格式。")

# Identify band columns = all columns except 'type'
band_cols = [c for c in df.columns if c != 'type']
if len(band_cols) != n_bands_expected:
    raise RuntimeError(f"样点表波段列数为 {len(band_cols)}，但期望 {n_bands_expected}。请确认样点文件列顺序（第一列为 type，其余按波段顺序）。")

# Fetch columns for B61, B112, B141 (1-based -> index-1)
col_B61  = band_cols[B61 - 1]
col_B112 = band_cols[B112 - 1]
col_B141 = band_cols[B141 - 1]

print("Using sample columns:", col_B61, col_B112, col_B141)

B61_vals  = df[col_B61].astype(float).values
B112_vals = df[col_B112].astype(float).values
B141_vals = df[col_B141].astype(float).values

# Impute missing values (fit on samples)
imputer = SimpleImputer(strategy='mean')
stacked_samples = np.vstack([B61_vals, B112_vals, B141_vals]).T
stacked_imp = imputer.fit_transform(stacked_samples)
B61_vals_imp  = stacked_imp[:,0]
B112_vals_imp = stacked_imp[:,1]
B141_vals_imp = stacked_imp[:,2]

pwadai_samples = compute_pwadai(B61_vals_imp, B112_vals_imp, B141_vals_imp)

# Use only type 0/1 samples to compute threshold and evaluate (paper uses PWDAI for binary)
mask01 = np.isin(df['type'].values, [0,1])
if mask01.sum() == 0:
    raise RuntimeError("样点中未包含 type 0 或 1，无法进行二分类 PWDAI 评估。")

pwadai_01 = pwadai_samples[mask01]
labels_01 = df['type'].values[mask01].astype(int)

# Compute Otsu threshold
thresh = threshold_otsu(pwadai_01)
print("Otsu threshold computed from samples:", thresh)

# Evaluate on sample subset
pred_samples = (pwadai_01 > thresh).astype(int)
print("\n=== Sample evaluation (PWDAI + Otsu) ===")
print(classification_report(labels_01, pred_samples, digits=4))
print("Confusion matrix:\n", confusion_matrix(labels_01, pred_samples))
print("Accuracy:", accuracy_score(labels_01, pred_samples))

# ---------------- 2) Apply to full image (block-wise) ----------------
print("\nProcessing image:", image_tif)
with rasterio.open(image_tif) as src:
    profile = src.profile.copy()
    num_bands = src.count
    height = src.height
    width = src.width
    src_nodata = src.nodata

    # check required bands exist
    max_band_needed = max(B61, B112, B141)
    if num_bands < max_band_needed:
        raise RuntimeError(f"影像波段数 {num_bands} < 需要的最大波段 {max_band_needed}. 请确认影像波段与样点一致。")

    profile.update(count=1, dtype=rasterio.uint8, nodata=nodata_value)
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(output_tif, 'w', **profile) as dst:
        for row_off in range(0, height, block_rows):
            h = min(block_rows, height - row_off)
            window = Window(col_off=0, row_off=row_off, width=width, height=h)
            block = src.read(window=window)  # shape (bands, h, width)

            # background mask: use src.nodata if present, else treat all-zero as background
            if src_nodata is not None:
                valid_mask = np.any(block != src_nodata, axis=0)  # (h, w)
            else:
                valid_mask = ~np.all(block == 0, axis=0)

            pix = block.reshape(num_bands, -1).T  # shape (h*w, bands)
            valid_flat = valid_mask.flatten()

            out_flat = np.full((h*width,), nodata_value, dtype=np.uint8)

            if valid_flat.sum() > 0:
                # extract bands (pix columns are 0-based)
                B61_img  = pix[valid_flat, B61  - 1].astype(float)
                B112_img = pix[valid_flat, B112 - 1].astype(float)
                B141_img = pix[valid_flat, B141 - 1].astype(float)

                stacked_img = np.vstack([B61_img, B112_img, B141_img]).T
                stacked_img_imp = imputer.transform(stacked_img)  # use sample-fitted imputer

                B61_img_imp  = stacked_img_imp[:,0]
                B112_img_imp = stacked_img_imp[:,1]
                B141_img_imp = stacked_img_imp[:,2]

                pwadai_img = compute_pwadai(B61_img_imp, B112_img_imp, B141_img_imp)
                cls = (pwadai_img > thresh).astype(np.uint8)  # 0 or 1
                out_flat[valid_flat] = cls

            out_block = out_flat.reshape(h, width)
            dst.write(out_block, 1, window=window)

print("Saved PWDAI classification to:", output_tif)

# ---------------- 3) Visualize (mask nodata) and print class stats ----------------
with rasterio.open(output_tif) as dst:
    arr = dst.read(1)

masked = np.ma.masked_equal(arr, nodata_value)
plt.figure(figsize=(6, 10))
plt.imshow(masked, cmap=ListedColormap(['green','red']))
plt.title(f"PWDAI classification (0=green,1=red)  threshold={thresh:.6f}")
plt.axis('off')
plt.show()

valid_mask_total = (arr != nodata_value)
unique, counts = np.unique(arr[valid_mask_total], return_counts=True)
total_valid = valid_mask_total.sum()
print("\nPixel counts (valid only):")
for u,c in zip(unique, counts):
    print(f"Class {u}: {c} pixels, {c/total_valid*100:.2f}%")
