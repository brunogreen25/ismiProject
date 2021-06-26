import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import json

# NOTE: Before running, move out of this directory and into "Train data" directory

def convert_dict_to_np_arrays(dict_):
    dict_keys = np.array(list(dict_.keys()))
    dict_values = np.array(list(dict_.values()))

    return dict_keys, dict_values

def add_to_dict(dict_, element):
    for row in element:
        for pixel in row:
            if pixel[0] in dict_['r'].keys():
                dict_['r'][pixel[0]] += 1
            else:
                dict_['r'][pixel[0]] = 1

            if pixel[1] in dict_['g'].keys():
                dict_['g'][pixel[1]] += 1
            else:
                dict_['g'][pixel[1]] = 1

            if pixel[2] in dict_['b'].keys():
                dict_['b'][pixel[2]] += 1
            else:
                dict_['b'][pixel[2]] = 1


#region METADATA
annotation_file = 'MIDOG.json'
img_metadata = {}

with open(annotation_file) as f:
    data = json.load(f)

    for img in data['images']:
        img_metadata[img['id']] = {
            'filename': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    for annotation in data['annotations']:
        info = {
            'id': annotation['id'],
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox']
        }

        try:
            img_metadata[annotation['image_id']]['labels'].append(info)
        except Exception:

            img_metadata[annotation['image_id']]['labels'] = []
            img_metadata[annotation['image_id']]['labels'].append(info)
#endregion

folder_path = 'images/*.tiff'
bar_width = 0.05

for i, img_path in enumerate(glob.glob(folder_path)):
    # Load image and its metadata
    image = mpimg.imread(img_path)
    img_id = i + 1
    img_meta = img_metadata[img_id]

    # Reset the counter when you come to new scanner
    if img_id == 1 or img_id == 51 or img_id == 101:
        mitosis_pixels = {'r': {}, 'g': {}, 'b': {}}
        hard_negs_pixels = {'r': {}, 'g': {}, 'b': {}}

    # If there are no more labels, there is nothing to calculate
    if 'labels' not in img_meta.keys():
        break

    # Calc vals for every regular pixel
    for label in img_meta['labels']:
        x0, y0, x1, y1 = label['bbox']
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        obj = image[x0:x1][y0:y1]
        try:
            if label['category_id'] == 1:
                add_to_dict(mitosis_pixels, obj)
            else:
                add_to_dict(hard_negs_pixels, obj)
        except Exception:
            print(f"Exception at id: {img_id}")
            print(len(mitosis_pixels), len(hard_negs_pixels))

    # Calc stats for every scanner
    if img_id == 50 or img_id == 100 or img_id == 150:
        scanner_id = int(img_id/50)
        print(f"Scanner id: {scanner_id}")
        # Histograms
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

        # Plot mitosis
        red_keys, red_vals = convert_dict_to_np_arrays(mitosis_pixels['r'])
        green_keys, green_vals = convert_dict_to_np_arrays(mitosis_pixels['g'])
        blue_keys, blue_vals = convert_dict_to_np_arrays(mitosis_pixels['b'])
        ax0.bar(red_keys - bar_width, red_vals, width=bar_width, color='r', align='center')
        ax0.bar(green_keys, green_vals, width=bar_width, color='g', align='center')
        ax0.bar(blue_keys + bar_width, blue_vals, width=bar_width, color='b', align='center')
        ax0.set_title("Mitosis color distribution")
        ax0.legend(['red', 'green', 'blue'])

        # Plot Hard Negs
        red_keys, red_vals = convert_dict_to_np_arrays(hard_negs_pixels['r'])
        green_keys, green_vals = convert_dict_to_np_arrays(hard_negs_pixels['g'])
        blue_keys, blue_vals = convert_dict_to_np_arrays(hard_negs_pixels['b'])
        ax1.bar(red_keys - bar_width, red_vals, width=bar_width, color='r', align='center')
        ax1.bar(green_keys, green_vals, width=bar_width, color='g', align='center')
        ax1.bar(blue_keys + bar_width, blue_vals, width=bar_width, color='b', align='center')
        ax1.set_title("Hard negatives color distribution")
        ax1.legend(['red', 'green', 'blue'])

        plt.savefig(f"stats_histograms/scanner_{scanner_id}.png")

        # Min, Max, Range, Mode, IQR and Median
        for color, dict_ in mitosis_pixels.items():
            mitosis_vals = list(dict_.keys())
            print(f"STATS: mitosis, {color}")

            # Max, min, range
            mitosis_max, mitosis_min = max(mitosis_vals), min(mitosis_vals)
            print(f"Mitosis max: {mitosis_max}, min: {mitosis_min}, range: {mitosis_max - mitosis_min}")

            # Mode
            mitosis_mode = max(set(mitosis_vals), key=mitosis_vals.count)
            print(f"Mitosis mode: {mitosis_mode}")

            # IQR
            q3, q2, q1 = np.percentile(mitosis_vals, [75, 50, 25])
            mitosis_interquantile_range = q3 - q1
            print(f"Mitosis interquantile range: {mitosis_interquantile_range}")

            # Median
            mitosis_median = statistics.median(mitosis_vals)
            print(f"Mitosis median: {mitosis_median}")


        for color, dict_ in hard_negs_pixels.items():
            hard_negs_vals = list(dict_.keys())
            print(f"STATS: hard negatives, {color}")

            # Max, min, range
            hard_negs_max, hard_negs_min = max(hard_negs_vals), min(hard_negs_vals)
            print(f"Hard negs max: {hard_negs_max}, min: {hard_negs_min}, range: {hard_negs_max - hard_negs_min}")

            # Mode
            hard_negs_mode = max(set(hard_negs_vals), key=hard_negs_vals.count)
            print(f"Hard negs mode: {hard_negs_mode}")

            # IQR
            q3, q2, q1 = np.percentile(hard_negs_vals, [75, 50, 25])
            hard_negs_interquantile_range = q3 - q1
            print(f"Hard negs interquantile range: {hard_negs_interquantile_range}")

            # Median
            hard_negs_median = statistics.median(hard_negs_vals)
            print(f"Hard negs median: {hard_negs_median}")