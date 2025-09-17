
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import csv

IMG_SIZE = (256, 256)
N_CLASSES = 4  # background, water, riverbank, land
MODEL_PATH = 'unet_river_segmentation.h5'
TEST_IMG_DIR = 'data/dataset/test/images'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_mask(img_path):
    img = Image.open(img_path).resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)
    mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
    return mask

def calculate_row_widths(mask):
    # Returns width for each row (segment)
    widths = []
    for row in mask:
        water_pixels = np.where(row == 1)[0]
        if len(water_pixels) > 0:
            width = water_pixels[-1] - water_pixels[0] + 1
        else:
            width = 0
        widths.append(width)
    return np.array(widths)

# Predict and collect widths for all test images
year_to_widths = {}
for fname in sorted(os.listdir(TEST_IMG_DIR)):
    if fname.endswith('.tif'):
        year = fname[:4]  # assumes filename starts with year
        img_path = os.path.join(TEST_IMG_DIR, fname)
        mask = predict_mask(img_path)
        row_widths = calculate_row_widths(mask)
        year_to_widths[year] = row_widths
        # Save mask as image
        mask_img = Image.fromarray((mask * 60).astype(np.uint8))
        mask_img.save(os.path.join(RESULTS_DIR, fname.replace('.tif', '_mask.png')))

# Stack widths by year (sorted)
years = sorted(year_to_widths.keys())
width_matrix = np.stack([year_to_widths[y] for y in years])  # shape: (years, rows)

# Find 4 rows (segments) with most width change
diff = width_matrix.max(axis=0) - width_matrix.min(axis=0)
top4_idx = np.argsort(diff)[-4:][::-1]  # indices of 4 rows with largest change

# Save segment width changes
top4_results = []
for idx in top4_idx:
    seg_widths = width_matrix[:, idx]
    for i, year in enumerate(years):
        top4_results.append([year, f'segment_{idx}', seg_widths[i]])

with open(os.path.join(RESULTS_DIR, 'top4_segment_widths.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['year', 'segment_row', 'width_pixels'])
    writer.writerows(top4_results)

print('Top 4 segments with most width change saved to results/top4_segment_widths.csv')
print('Segment rows (from top):', top4_idx)
max_change = row_changes[max_change_row]

# Report
print(f"Segment (row) with most width change: {max_change_row}")
print(f"Max width change across years: {max_change:.2f} pixels")
for i, year in enumerate(all_years):
    print(f"{year}: width at row {max_change_row} = {width_matrix[i, max_change_row]:.2f} pixels")

# Save summary to CSV
with open(os.path.join(RESULTS_DIR, 'river_widths_by_row.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['year'] + [f'row_{i}' for i in range(IMG_SIZE[0])])
    for i, year in enumerate(all_years):
        writer.writerow([year] + list(width_matrix[i]))

with open(os.path.join(RESULTS_DIR, 'max_change_segment.txt'), 'w') as f:
    f.write(f'Segment (row) with most width change: {max_change_row}\n')
    f.write(f'Max width change across years: {max_change:.2f} pixels\n')
    for i, year in enumerate(all_years):
        f.write(f'{year}: width at row {max_change_row} = {width_matrix[i, max_change_row]:.2f} pixels\n')

print('Results saved to results/river_widths_by_row.csv and results/max_change_segment.txt')
