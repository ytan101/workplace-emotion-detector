
import os, textwrap
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageStat


def get_image_features(img_path, emotion_class):
    """ Get a list of features from a single image, in this order:
        1. emotion_class
        2. approx_brightness
        3. pixel_std
        4. percent_distinct_pix
        5. pixel_values

    Args:
        img_path (str): Path to image
        img_class (int): emotion_class e.g. 0: 'angry'

    Returns:
        List: List of features
    """
    img_pil = Image.open(img_path)

    # Get pixel values
    pixel_values = list(img_pil.getdata())
    
    # Get approximate image brightness
    stat = ImageStat.Stat(img_pil)
    approx_brightness = stat.rms[0] 

    # Get image pixels std
    pixel_std = stat.stddev[0]

    # Get image pixels median
    pixel_median = stat.median[0]

    # Get percentage of n_distinct_pixels over n_all_possible_values i.e. 256
    percent_distinct_pix = len(set(pixel_values)) / 256

    return [emotion_class, approx_brightness, pixel_median, 
            pixel_std, percent_distinct_pix, 
            img_path, pixel_values]


def convert_image_paths_to_pil(img_paths_list):
    """ Takes image paths and convert to PIL images

    Args:
        img_paths_list (str): List of image paths

    Returns:
        List: List of PIL images
    """
    img_pil_list=[]

    for img_path in img_paths_list:
        img_pil_list.append(Image.open(img_path))

    return img_pil_list


def display_greyscale_images(
    images_pil_list, 
    columns=5, width=20, height=8, max_images=120, 
    label_wrap_length=50, label_font_size=8):
    """ Display greyscaled images in a grid

    Args:
        images_pil_list (List): Greyscale PIL images to display
        columns (int, optional): Defaults to 5.
        width (int, optional): Width of grid. Defaults to 20.
        height (int, optional): Height of grid. Defaults to 8.
        max_images (int, optional): Defaults to 120.
        label_wrap_length (int, optional): Defaults to 50.
        label_font_size (int, optional): Defaults to 8.
    """
    if not images_pil_list:
        print("No images to display.")
        return 

    if len(images_pil_list) > max_images:
        print(f"Showing {max_images} images of {len(images_pil_list)}:")
        images_pil_list=images_pil_list[0:max_images]

    height = max(height, int(len(images_pil_list)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images_pil_list):

        plt.subplot(int(len(images_pil_list) / columns + 1), columns, i + 1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

