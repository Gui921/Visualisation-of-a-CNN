import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk, ImageDraw

def resize_image(img, target_size):
    return img.resize(target_size, Image.LANCZOS)

def create_combined_image_with_lines(filter_paths, pred_path, fmap_paths, output_path, target_size=(200, 200), line_color=(0, 0, 0), line_width=2):

    # Open all images and resize them
    filter_images = [resize_image(Image.open(path), target_size) for path in filter_paths]
    pred_image = resize_image(Image.open(pred_path), target_size)
    fmap_images = [resize_image(Image.open(path), target_size) for path in fmap_paths]

    # Calculate dimensions for the final image
    max_filter_width = max([img.width for img in filter_images])
    total_width = max_filter_width + pred_image.width + max([img.width for img in fmap_images])
    max_height = max(max([img.height for img in filter_images]), pred_image.height, max([img.height for img in fmap_images]))

    # Create a new image
    combined_image = Image.new('RGB', (total_width + 400, 1800), (255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    # Paste filter images on the left and draw lines to pred.png
    current_height = 400
    for img in filter_images:
        combined_image.paste(img, (0, current_height))
        filter_center = (img.width, current_height + img.height // 2)
        pred_center = (max_filter_width + 200, 900 + img.height /2)
        draw.line([filter_center, pred_center], fill=line_color, width=line_width)
        current_height += img.height

    # Paste pred.png in the center and draw lines to fmaps
    pred_x = max_filter_width + 200
    pred_y = (max_height - pred_image.height) // 2
    combined_image.paste(pred_image, (pred_x, 900))
    pred_center = (pred_x + pred_image.width, 900 + pred_image.height // 2)

    # Paste fmap images on the right and draw lines to pred.png
    current_height = 400
    fmap_x = pred_x + pred_image.width + 200
    for img in fmap_images:
        combined_image.paste(img, (fmap_x, current_height))
        fmap_center = (fmap_x, current_height + img.height // 2)
        draw.line([pred_center, fmap_center], fill=line_color, width=line_width)
        current_height += img.height

    # Save the combined image
    combined_image.save(output_path)

def create_combined_image_with_lines_2(filter_paths, filter_paths_2, pred_path, fmap_paths,fmap_paths_2 ,output_path, target_size=(200, 200), line_color=(0, 0, 0), line_width=2):

    # Open all images and resize them
    filter_images = [resize_image(Image.open(path), target_size) for path in filter_paths]
    filter_images2 = [resize_image(Image.open(path), target_size) for path in filter_paths_2]
    pred_image = resize_image(Image.open(pred_path), target_size)
    fmap_images = [resize_image(Image.open(path), target_size) for path in fmap_paths]
    fmap_images2 = [resize_image(Image.open(path), target_size) for path in fmap_paths_2]

    # Calculate dimensions for the final image
    max_filter_width = max([img.width for img in filter_images])
    max_fmap_width = max([img.width for img in fmap_images])
    max_filter_width2 = max([img.width for img in filter_images2])
    total_width = max_filter_width + pred_image.width + max_fmap_width + max_filter_width2 +max([img.width for img in fmap_images2]) + 400
    max_height = max(max([img.height for img in filter_images]), pred_image.height, max([img.height for img in fmap_images]), max([img.height for img in fmap_images2]))

    # Create a new image
    combined_image = Image.new('RGB', (total_width + 400, 1800), (255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    # Paste filter images on the left and draw lines to pred.png
    current_height = 400
    for img in filter_images:
        combined_image.paste(img, (0, current_height))
        filter_center = (img.width, current_height + img.height // 2)
        pred_center = (max_filter_width + 200, 900 + img.height /2)
        draw.line([filter_center, pred_center], fill=line_color, width=line_width)
        current_height += img.height

    # Paste pred.png in the center and draw lines to fmaps
    pred_x = max_filter_width + 200
    pred_y = (max_height - pred_image.height) // 2
    combined_image.paste(pred_image, (pred_x, 900))
    pred_center = (pred_x + pred_image.width, 900 + pred_image.height // 2)

    # Paste fmap images on the right and draw lines to pred.png
    current_height = 400
    fmap_x = pred_x + pred_image.width + 200
 
    for img in fmap_images:
        combined_image.paste(img, (fmap_x, current_height))
        fmap_center = (fmap_x, current_height + img.height // 2)
        draw.line([pred_center, fmap_center], fill=line_color, width=line_width)
        current_height += img.height
    
    #filter_2
    current_height = 400
    filter_x_2 = fmap_x + pred_image.width + 200
    for img in filter_images2:
        combined_image.paste(img,(filter_x_2,current_height))
        filter_center_2 = (filter_x_2, current_height + img.height // 2)
        draw.line([(filter_x_2 - 200, current_height + img.height // 2), filter_center_2], fill=line_color, width=line_width)
        current_height += img.height


    # fmap_2
    current_height = 400
    fmap_x_2 = filter_x_2 + pred_image.width + 200
    for img in fmap_images2:
        combined_image.paste(img,(fmap_x_2,current_height))
        fmap_center_2 = (fmap_x_2, current_height + img.height // 2)
        draw.line([(fmap_x_2 - 200, current_height + img.height // 2), fmap_center_2], fill=line_color, width=line_width)
        current_height += img.height

    # Save the combined image
    combined_image.save(output_path)

def get_combined_image(model_type):
    filter_paths = ['filters/filter 0.png', 'filters/filter 1.png', 'filters/filter 2.png', 'filters/filter 3.png', 'filters/filter 4.png', 'filters/filter 5.png']
    pred_path = 'pred.png'
    fmap_paths = ['fmaps/fmap 0.png', 'fmaps/fmap 1.png', 'fmaps/fmap 2.png', 'fmaps/fmap 3.png', 'fmaps/fmap 4.png', 'fmaps/fmap 5.png']
    output_path = 'combined_image.png'

    if model_type == 1:
        create_combined_image_with_lines(filter_paths, pred_path, fmap_paths, output_path)
    else:
        filter_paths_2 = ['filters/filter 0_2.png', 'filters/filter 1_2.png', 'filters/filter 2_2.png', 'filters/filter 3_2.png', 'filters/filter 4_2.png', 'filters/filter 5_2.png']
        fmap_paths_2 = ['fmaps/fmap 0_2.png', 'fmaps/fmap 1_2.png', 'fmaps/fmap 2_2.png', 'fmaps/fmap 3_2.png', 'fmaps/fmap 4_2.png', 'fmaps/fmap 5_2.png']
        create_combined_image_with_lines_2(filter_paths,filter_paths_2, pred_path, fmap_paths,fmap_paths_2, output_path)
