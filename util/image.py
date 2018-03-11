import os

import cv2
import numpy as np


def find_image_files(input_dir):
    """
    Recursively searches for .jpg/.png images.
    """
    for root_dir, found_dirs, found_files in os.walk(input_dir):
        for found_file in found_files:
            if os.path.splitext(found_file)[-1].lower() in ['.jpg', '.png']:
                yield os.path.join(root_dir, found_file)


def load_image(input_file,
               output_color_space='RGB'):
    """
    Load image, normalize, and convert color space.
    """
    if not os.path.exists(input_file):
        raise IOError(input_file + ' does not exist or could not be loaded')
    input_image = cv2.imread(input_file).astype(np.float32) / 255
    if output_color_space == 'BGR':
        output_image = input_image
    elif output_color_space == 'RGB':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    elif output_color_space == 'GRAY':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif output_color_space == 'HSV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    elif output_color_space == 'LUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LUV)
    elif output_color_space == 'YUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    elif output_color_space == 'YCrCb':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
    else:
        raise IOError('invalid color_space: {}'.format(output_color_space))
    return output_image


def save_image(input_image,
               output_path,
               output_type=None,
               input_color_space='RGB'):
    """
    Convert color space, de-normalize and save image.
    """
    if input_image.dtype != np.uint8:
        input_image = (input_image * 255).astype(np.uint8)

    if input_color_space == 'BGR':
        output_image = input_image
    elif input_color_space == 'RGB':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    elif input_color_space == 'GRAY':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    elif input_color_space == 'HSV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_HSV2BGR)
    elif input_color_space == 'LUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_LUV2BGR)
    elif input_color_space == 'YUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_YUV2BGR)
    elif input_color_space == 'YCrCb':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_YCrCb2BGR)
    else:
        raise IOError('invalid color_space: {}'.format(input_color_space))

    # Create directories
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    output_base, output_ext = os.path.splitext(output_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Save files
    if output_type is None:
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(output_base)), output_image)
    else:
        type_dir = '{}/by_type/{}'.format(output_dir, output_type)
        if not os.path.isdir(type_dir):
            os.makedirs(type_dir)
        name_dir = '{}/by_name/{}'.format(output_dir, output_base)
        if not os.path.isdir(name_dir):
            os.makedirs(name_dir)

        cv2.imwrite(
            os.path.join(type_dir, '{}_{}.png'.format(output_base, output_type)),
            output_image)
        cv2.imwrite(
            os.path.join(name_dir, '{}_{}.png'.format(output_base, output_type)),
            output_image)


def get_image_histogram(input_image, histogram_bins=32, bins_range=(0, 256)):
    """
    Builds image histogram.
    """
    return np.concatenate(
        [np.histogram(input_image[:, :, input_channel], histogram_bins, bins_range)[0]
         for input_channel in range(input_image.shape[-1])]) / (input_image.shape[0] * input_image.shape[1])


def resize_and_vectorize_image(input_image, output_size=(32, 32)):
    """
    Scale and vectorize image.
    """
    return cv2.resize(input_image, output_size).ravel()


def convert_video_colorspace(input_frame,
                             output_color_space='RGB'):
    """
    Convert video frame color space.
    """
    if input_frame.dtype != np.float32:
        input_frame = input_frame.astype(np.float32) / 255

    if output_color_space == 'RGB':
        output_frame = input_frame
    elif output_color_space == 'GRAY':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    elif output_color_space == 'BGR':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
    elif output_color_space == 'HSV':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2HSV)
    elif output_color_space == 'LUV':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2LUV)
    elif output_color_space == 'YUV':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2YUV)
    elif output_color_space == 'YCrCb':
        output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2YCrCb)
    else:
        raise IOError('invalid color_space: {}'.format(output_color_space))

    return output_frame


def convert_image_to_rgb(input_image,
                         input_color_space):
    """
    Convert image color space.
    """
    if input_color_space == 'GRAY':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_color_space == 'BGR':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    elif input_color_space == 'HSV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_HSV2RGB)
    elif input_color_space == 'LUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_LUV2RGB)
    elif input_color_space == 'YUV':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_YUV2RGB)
    elif input_color_space == 'YCrCb':
        output_image = cv2.cvtColor(input_image, cv2.COLOR_YCrCb2RGB)
    else:
        raise IOError('invalid color_space: {}'.format(input_color_space))

    if output_image.dtype != np.uint8:
        output_image = (output_image * 255).astype(np.uint8)

    return output_image
