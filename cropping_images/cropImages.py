import cv2
import json
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set locations
midog_location = 'MIDOG.json'
new_midog_location = 'MIDOG_new.json'
source_location = 'TrainStainData/images'
destination_location = 'TrainStainCropData/images'

# Read MIDOG.json and instantiate new json annotation
with open(midog_location) as f:
    annotation = json.load(f)
new_annotation = {
    'info': annotation['info'],
    'licenses': annotation['licenses'],
    'categories': annotation['categories'],
    'images': [],
    'annotations': []
}

# Set desired height, width
patch_height, patch_width = (500, 500)

# Get image metadata
image_metadata = {}
for img_meta in annotation['images']:
    image_metadata[img_meta['id']] = {
        'file_name': img_meta['file_name'],
        'height': img_meta['height'],
        'width': img_meta['width']
    }

# Get category metadata
categories = {}
for cat_meta in annotation['categories']:
    categories[cat_meta['id']] = {
        'name': cat_meta['name']
    }

def visualize(loaded_image, bounding_box):
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #im_pil = Image.fromarray(img)

    fig, ax = plt.subplots()
    ax.imshow(loaded_image)

    # Add bounding box
    xmin, ymin, xmax, ymax = bounding_box
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

# Parse annotations
prev_image_name = None
annotation_counter = 0
annotation_number = len(annotation['annotations'])
for idx, anno in enumerate(annotation['annotations']):

    # Read annotation
    bbox = anno['bbox']
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    category_id = anno['category_id']
    image_id = anno['image_id']

    # Get height and width of bounding box and image
    bbox_height = ymax - ymin
    bbox_width = xmax - xmin
    image_width = image_metadata[image_id]['width']
    image_height = image_metadata[image_id]['height']

    # Set annotation counter
    curr_image_name = image_metadata[image_id]['file_name']
    if idx != 0 and curr_image_name == prev_image_name:
        annotation_counter += 1
    else:
        annotation_counter = 0
    prev_image_name = curr_image_name

    # Load image
    loaded_image_path = os.path.join(source_location, image_metadata[image_id]['file_name'])
    loaded_image = cv2.imread(loaded_image_path)

    # Generate random crop to the left and bottom sides
    additional_patch_height = 0
    additional_patch_width = 0

    crop_xmin = random.randint(0, patch_width - bbox_width)
    crop_ymin = random.randint(0, patch_height - bbox_height)
    if xmin - crop_xmin <= 0:
        # Not enoguh space left
        if xmin >= 0:
            crop_xmin = random.randint(0, xmin)
        else:
            crop_xmin = 0
            additional_patch_width = -1 * xmin
            bbox_width += 1
            xmin = 0
    if ymin - crop_ymin <= 0:
        # Not enough space down
        if ymin >= 0:
            crop_ymin = random.randint(0, ymin)
        else:
            crop_ymin = 0
            additional_patch_height = -1 * ymin
            bbox_height += 1
            ymin = 0

    additional_patch_width = 0
    additional_patch_height = 0

    crop_xmax = patch_width - bbox_width - crop_xmin + additional_patch_width
    crop_ymax = patch_height - bbox_height - crop_ymin + additional_patch_height
    if xmax + crop_xmax >= image_width:
        # Not enough space right
        if xmax <= image_width:
            crop_xmax = random.randint(0, image_width-xmax)
        else:
            crop_xmax = 0
            additional_patch_width = -1 * (image_width - xmax)
            bbox_width -= 1
            xmax = image_width
        crop_xmin = patch_width - bbox_width - crop_xmax + additional_patch_width
    if ymax + crop_ymax >= image_height:
        # Not enough space up
        if ymax <= image_height:
            crop_ymax = random.randint(0, image_height-ymax)
        else:
            crop_ymax = 0
            additional_patch_height = -1 * (image_height - ymax)
            bbox_height -= 1
            ymax = image_height
        crop_ymin = patch_height - bbox_height - crop_ymax + additional_patch_height

    #region PRINT_ERROR_COMMENTED
    # Print error
    #print("HHHHHHH")
    #print(patch_width, bbox_height+crop_xmin+crop_xmax)
    #print(patch_height, bbox_height+crop_ymin+crop_ymax)
    #print(f'bbox width: {bbox_width}, BBox height: {bbox_height}')
    #print(f'BBox position on image: {xmin, ymin, xmax, ymax}')
    #print(f'image dim: {image_width, image_height}')
    #print(f'cropped: {crop_xmin, crop_ymin, crop_xmax, crop_ymax}')
    #print(idx)
    #endregion

    # Crop image
    cropped_image = loaded_image[ymin-crop_ymin: ymin-crop_ymin+patch_height, xmin-crop_xmin: xmin-crop_xmin+patch_width]
    cropped_bbox = [crop_xmin, crop_ymin, crop_xmin+bbox_width, crop_ymin+bbox_height]

    # Visualize cropped image
    #visualize(loaded_image, bbox)
    #visualize(cropped_image, cropped_bbox)

    # Save image
    new_file_name = image_metadata[image_id]['file_name'].split('.')[0] + '_' + str(annotation_counter) + '.tiff'
    cv2.imwrite('TrainStainCropData/images/'+ new_file_name, cropped_image)

    # Save new annotation
    new_annotation['images'].append({
        'license': 1,
        'file_name': new_file_name,
        'id': idx,
        'width': patch_width,
        'height': patch_height
    })
    new_annotation['annotations'].append({
        'id': idx,
        'category_id': category_id,
        'image_id': idx, # There is 1 annotation per image here
        'bbox': cropped_bbox,
        'crop_xmin': crop_xmin,
        'crop_max': crop_xmax,
        'crop_ymin': crop_ymin,
        'crop_ymax': crop_ymax
    })

    print(f"Cropped annotation {idx} out of {annotation_number}, percent: {round(idx/annotation_number*100, 2)}%")

# Save annotation file
with open(new_midog_location, 'w+') as f:
    json.dump(new_annotation, f)


