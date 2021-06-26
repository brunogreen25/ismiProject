import cv2
import os
import json

# Source and destination location
source_location = 'TestData/images/'
destination_location = 'TestCropData/images/'
img_paths = os.listdir(source_location)
img_number = len(img_paths)

metadata = {
    'images': [],
    'patches': []
}
patch_width, patch_height = (500, 500)

for idx, img_path in enumerate(img_paths):
    image_idx_str = str(idx) if idx > 10 else '0'+str(idx)

    loaded_image = cv2.imread(os.path.join(source_location, img_path))
    image_height, image_width, _ = loaded_image.shape

    crops_per_row = image_width // patch_width + 1
    crops_per_col = image_height // patch_height + 1

    # Save information to the annotation json
    metadata['images'].append({
        'file_name': img_path,
        'width': image_width,
        'height': image_height,
        'id': idx
    })

    # Without dilation (DILATION_IDX: 1=None, 2=x, 3=y)
    dilation_dict = {
        0: 'none',
        1: 'x',
        2: 'y'
    }
    dilation_x = 250
    dilation_y = 250
    for dilation_idx, (dilation_x, dilation_y) in enumerate([(0, 0), (0, dilation_x), (dilation_y, 0)]):
        for col_idx in range(crops_per_col):
            for row_idx in range(crops_per_row):
                patch_idx = col_idx * crops_per_col + row_idx
                patch_idx_str = str(patch_idx)

                xmin = col_idx * patch_width + dilation_x
                ymin = row_idx * patch_height + dilation_y
                xmax = col_idx * patch_width + patch_width + dilation_x
                ymax = row_idx * patch_height + patch_height + dilation_y

                if xmax > image_width:
                    # Last patch in the row
                    xmax = image_width
                    xmin = image_width - patch_width
                if ymax > image_height:
                    # Last patch in the column
                    ymax = image_height
                    ymin = image_height - patch_height

                # Crop the image
                cropped_image = loaded_image[ymin:ymax, xmin:xmax]

                # Save the cropped image
                full_path = destination_location + image_idx_str + '_' + patch_idx_str + '_' + dilation_dict[dilation_idx] + '.tiff'
                cv2.imwrite(full_path, cropped_image)

                # Save the annotation file
                metadata['patches'].append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'image_id': idx,
                    'id': patch_idx,
                    'dilation_id': dilation_idx,
                    'dilation_type': dilation_dict[dilation_idx]
                })

    print(f'Cropped image {image_idx_str} out of {img_number}; {idx/img_number*100}%')

# Save metadata
with open('test_metadata.json', 'w+') as f:
    json.dump(metadata, f)


