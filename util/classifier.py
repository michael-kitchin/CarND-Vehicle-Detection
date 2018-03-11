import cv2
import numpy as np
from skimage.feature import hog

from util import image


def get_hog_features(input_image,
                     orientation,
                     px_per_cell,
                     cell_per_blk,
                     visualize=False,
                     feature_vector=True):
    """
    Automates sklearn.hog().
    """
    return hog(input_image, orientations=orientation,
               pixels_per_cell=(px_per_cell, px_per_cell),
               cells_per_block=(cell_per_blk, cell_per_blk),
               transform_sqrt=True, visualise=visualize, feature_vector=feature_vector)


def extract_features_from_images(image_file_paths,
                                 color_space='RGB',
                                 image_size=(32, 32),
                                 histogram_bins=0,
                                 hog_orientation=9,
                                 hog_px_per_cell=8,
                                 hog_cell_per_blk=2,
                                 hog_channel='ALL'):
    """
    Loads images and extracts features.
    """
    output_features = []
    for image_path in image_file_paths:
        output_features.append(extract_features(image.load_image(image_path, color_space),
                                                image_size, histogram_bins,
                                                hog_orientation, hog_px_per_cell,
                                                hog_cell_per_blk, hog_channel))
    return np.array(output_features)


def extract_features(input_image,
                     image_size=(32, 32),
                     hist_bins=0,
                     hog_orientation=9,
                     hog_px_per_cell=8,
                     hog_cell_per_blk=2,
                     hog_channel='ALL'):
    """
    Extracts features from an image.
    """
    input_features = image.resize_and_vectorize_image(input_image, image_size)
    color_features = []
    if hist_bins > 0:
        color_features = image.get_image_histogram(input_image * 255, hist_bins)
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(input_image.shape[2]):
            hog_features.append(get_hog_features(input_image[:, :, channel],
                                                 hog_orientation, hog_px_per_cell, hog_cell_per_blk,
                                                 visualize=False, feature_vector=True))
        hog_features = np.ravel(hog_features)
    elif hog_channel == 'NONE':
        hog_features = []
    else:
        hog_features = get_hog_features(input_image[:, :, hog_channel],
                                        hog_orientation, hog_px_per_cell,
                                        hog_cell_per_blk, visualize=False,
                                        feature_vector=True)
    features = np.concatenate((input_features, color_features, hog_features))
    return features


def get_heat_map(input_shape, input_matches):
    """
    Builds heatmap from input shape and detected boxes.
    """
    output_heat_map = np.zeros(input_shape, dtype=np.float32)
    for match in input_matches:
        match = match[0], match[1]
        if len(match) > 2:
            hits = match[2]
        else:
            hits = 1
        output_heat_map[match[0][1]:match[1][1], match[0][0]:match[1][0]] += hits
    return output_heat_map


def draw_boxes(input_image,
               input_bboxes,
               line_color=(0, 0, 255),
               line_width=5):
    """
    Draws boxes on image.
    """
    output_image = np.copy(input_image)
    for bb in input_bboxes:
        if len(bb) > 2:
            line_color = np.array(line_color) * bb[2]
        cv2.rectangle(output_image, bb[0], bb[1], line_color, line_width)
    return output_image
