import argparse
import glob
import json
import os
import pickle

import cv2
import imageio
import numpy as np
from scipy.ndimage.measurements import label as label_image
from skimage.filters.rank import windowed_histogram

from util import image, classifier, running_mean

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

# Default window scales
default_window_scales = """[
    [0.5, 0.75, [0.0, 1.0], [0.5, 0.875]],
    [0.5714, 0.75, [0.0, 1.0], [0.5, 0.875]],
    [1.0, 0.5, [0.3333, 0.6666], [0.55, 0.9]],
    [1.3333, 0.5, [0.0, 1.0], [0.5, 0.75]],
    [2.0, 0.0, [0.25, 0.75], [0.55, 0.64]]
]"""

parser = argparse.ArgumentParser()

# Input/output files
parser.add_argument('--input-path', type=str, default='./project_video.mp4')
parser.add_argument('--output-path', type=str, default='./output_images/project_video.mp4')
parser.add_argument('--input-classifier-file', type=str, default='./classifier_data.p')

# Input video range (optional)
parser.add_argument('--input-video-range', type=str, default='[]')

# Other params
parser.add_argument('--window-scales', type=str, default=default_window_scales)
parser.add_argument('--window-min-score', type=float, default=1.5)
parser.add_argument('--match-threshold', type=float, default=1.0)
parser.add_argument('--match-min-size', type=str, default='[16,16]')
parser.add_argument('--match-average-frames', type=int, default=20)
parser.add_argument('--match-average-prune', type=int, default=100)

args = parser.parse_args()
print('Args: {}'.format(args))

# Transliterate params
input_path = args.input_path
output_path = args.output_path
input_classifier_file = args.input_classifier_file
input_video_range = tuple(json.loads(args.input_video_range))
window_scales = tuple(json.loads(args.window_scales))
window_min_score = args.window_min_score
match_threshold = args.match_threshold
match_min_size = tuple(json.loads(args.match_min_size))
match_average_frames = args.match_average_frames
match_average_prune = args.match_average_prune

with open(input_classifier_file, 'rb') as classifier_file:
    classifier_data = pickle.load(classifier_file)
    classifier_args = classifier_data['args']
    input_scaler = classifier_data['scaler']
    input_classifier = classifier_data['classifier']

print('Classifier args: {}'.format(classifier_args))

color_space = classifier_args.color_space
image_size = tuple(json.loads(classifier_args.image_size))
hog_size = tuple(json.loads(classifier_args.hog_size))
histogram_bins = classifier_args.histogram_bins
hog_orientation = classifier_args.hog_orientation
hog_px_per_cell = classifier_args.hog_px_per_cell
hog_cell_per_blk = classifier_args.hog_cell_per_blk

# May be 'ALL', 'NONE', or integer
try:
    hog_channel = int(classifier_args.hog_channel)
except ValueError:
    hog_channel = classifier_args.hog_channel


def get_image_hits(input_image,
                   hog_size=(64, 64),
                   px_per_cell=8,
                   min_score=1.0,
                   scales=None):
    """
    Finds & vectorizes HOG features in supplied image.
    """
    x_cells_per_window = hog_size[1] // px_per_cell - 1
    y_cells_per_window = hog_size[0] // px_per_cell - 1

    output_hits = []
    image_scale_x = image_size[0] / hog_size[0]
    image_scale_y = image_size[1] / hog_size[1]

    for scale, overlap, x_range, y_range in scales:
        # Subset image
        target_x = (int(x_range[0] * input_image.shape[1]), int(x_range[1] * input_image.shape[1]))
        target_y = (int(y_range[0] * input_image.shape[0]), int(y_range[1] * input_image.shape[0]))
        target_part = input_image[target_y[0]:target_y[1], target_x[0]:target_x[1], :]

        # Scale subset
        target_shape = (int(target_part.shape[1] * scale), int(target_part.shape[0] * scale))
        scaled_target = cv2.resize(target_part, target_shape)

        # Extract HOG features
        if hog_channel == 'ALL':
            hog_parts = [classifier.get_hog_features(scaled_target[:, :, channel],
                                                     orientation=hog_orientation,
                                                     px_per_cell=hog_px_per_cell,
                                                     cell_per_blk=hog_cell_per_blk,
                                                     feature_vector=False)
                         for channel in range(scaled_target.shape[-1])]
        else:
            hog_parts = [classifier.get_hog_features(scaled_target[:, :, hog_channel],
                                                     orientation=hog_orientation,
                                                     px_per_cell=hog_px_per_cell,
                                                     cell_per_blk=hog_cell_per_blk,
                                                     feature_vector=False)]
        hog_shape = hog_parts[0].shape

        # Get histograms
        image_histogram = \
            [windowed_histogram((scaled_target[:, :, channel] * 255.0 / 256.0 * histogram_bins).astype(np.uint8),
                                selem=np.ones(hog_size),
                                shift_x=-hog_size[1] // 2,
                                shift_y=-hog_size[0] // 2,
                                n_bins=histogram_bins)
             for channel in range(scaled_target.shape[-1])]

        #
        scaled_shape = (int(target_shape[0] * image_scale_y),
                        int(target_shape[1] * image_scale_x))
        scaled_image = cv2.resize(scaled_target, scaled_shape)

        # X/Y ranges for walking features
        x_start = 0
        x_stop = hog_shape[1] - x_cells_per_window + 1
        x_step = int((1 - overlap) * x_cells_per_window)

        y_start = 0
        y_stop = hog_shape[0] - y_cells_per_window + 1
        y_step = int((1 - overlap) * y_cells_per_window)

        for x_px in range(x_start, x_stop, x_step):
            for y_px in range(y_start, y_stop, y_step):
                feature_start_x = int(x_px * px_per_cell * image_scale_x)
                feature_end_x = feature_start_x + image_size[0]

                feature_start_y = int(y_px * px_per_cell * image_scale_y)
                feature_end_y = feature_start_y + image_size[1]

                # Base image vector
                image_features = scaled_image[feature_start_y:feature_end_y, feature_start_x:feature_end_x, :].ravel()

                # Color feature vector
                color_features = np.ravel(
                    [histogram_bin[(y_px * px_per_cell), (x_px * px_per_cell), :].ravel()
                     for histogram_bin in image_histogram])

                # Hog feature vector
                hog_features = np.ravel(
                    [hog_part[y_px:y_px + y_cells_per_window, x_px:x_px + x_cells_per_window].ravel()
                     for hog_part in hog_parts])

                # Buidl window
                window_start = (target_x[0] + int(x_px / scale * px_per_cell),
                                target_y[0] + int(y_px / scale * px_per_cell))
                window_end = (int(window_start[0] + hog_size[1] / scale),
                              int(window_start[1] + hog_size[0] / scale))

                # Vectorize all features
                all_features = np.concatenate((image_features, color_features, hog_features))
                all_features = all_features.reshape(1, -1)
                all_features = input_scaler.transform(all_features)

                # Score & check
                classifier_score = input_classifier.decision_function(all_features)
                if classifier_score >= min_score:
                    output_hits.append((window_start, window_end, scale ** 2))

    return output_hits


def process_image(input_image,
                  mean_heatmap=None,
                  hog_size=(64, 64),
                  px_per_cell=8,
                  min_score=1.0,
                  scales=None,
                  threshold=1.0,
                  min_size=(16, 16)):
    """
    Finds and marks matching objects in supplied image.
    """
    hit_ctr = get_image_hits(input_image,
                             hog_size=hog_size,
                             px_per_cell=px_per_cell,
                             min_score=min_score,
                             scales=scales)
    heat_map = classifier.get_heat_map(input_image.shape[0:2], hit_ctr)

    if not mean_heatmap is None:
        heat_map = mean_heatmap.update_mean(heat_map)

    binary_map = heat_map >= threshold
    labels = label_image(binary_map)

    found_matches = []
    for i in range(labels[1]):
        y_points, x_points = np.where(labels[0] == i + 1)
        match = ((np.min(x_points), np.min(y_points)),
                 (np.max(x_points), np.max(y_points)))
        width_px = match[1][0] - match[0][0]
        height_px = match[1][1] - match[0][1]

        if width_px >= min_size[0] and height_px >= min_size[1]:
            found_matches.append(match)

    return found_matches


# Search for usable files.
for input_file in glob.glob(input_path):
    file_extension = os.path.splitext(input_file)[-1].lower()
    output_file = os.path.join(output_path, os.path.basename(input_file))

    if file_extension in ['.jpg', '.png']:

        print('Loading {} (image)...'.format(input_file))
        input_image = image.load_image(input_file, color_space)
        print('...{} (image) loaded.'.format(input_file))

        print('Detecting vehicles...')
        found_matches = process_image(input_image, mean_heatmap=None,
                                      hog_size=hog_size, px_per_cell=hog_px_per_cell,
                                      min_score=window_min_score, scales=window_scales,
                                      threshold=match_threshold, min_size=match_min_size)
        print('...Vehicles detected: {}'.format(found_matches if len(found_matches) > 0 else '(none)'))
        output_image = classifier.draw_boxes(image.convert_image_to_rgb(input_image, color_space),
                                             found_matches)

        print('Saving {} (image)...'.format(output_file))
        image.save_image(output_image, output_file)
        print('...{} (image) saved.'.format(output_file))

    elif file_extension in ['.mp4']:
        mean_heatmap = \
            running_mean.RunningMean(max_size=match_average_frames, max_prunes=match_average_prune)


        def process_frame(input_frame):
            work_frame = image.convert_video_colorspace(input_frame, color_space)
            found_matches = process_image(work_frame, mean_heatmap=mean_heatmap,
                                          hog_size=hog_size, px_per_cell=hog_px_per_cell,
                                          min_score=window_min_score, scales=window_scales,
                                          threshold=match_threshold, min_size=match_min_size)
            output_frame = classifier.draw_boxes(input_frame,
                                                 found_matches)
            return output_frame


        video_clip = VideoFileClip(input_file)

        if input_video_range and len(input_video_range) == 2:
            video_clip = video_clip.subclip(input_video_range[0], input_video_range[1])
        video_clip = video_clip.fl_image(process_frame)

        print('Saving {} (video)...'.format(output_file))
        video_clip.write_videofile(output_file, audio=False)
        print('...{} (video) saved.'.format(output_file))
    else:
        raise IOError('invalid file extensoin: {}' + file_extension)
