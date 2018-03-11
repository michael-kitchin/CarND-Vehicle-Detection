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
    [0.5, 0.75, [0.0, 1.0], [0.5, 0.9]],
    [0.6, 0.75, [0.0, 1.0], [0.5, 0.9]],
    [1.0, 0.5, [0.3333, 0.6666], [0.55, 0.9]],
    [1.3333, 0.5, [0.0, 1.0], [0.5, 0.75]],
    [2.0, 0.0, [0.25, 0.75], [0.55, 0.65]]
]"""

parser = argparse.ArgumentParser()

# Input/output files
parser.add_argument('--input-path', type=str, default='./*.mp4')
parser.add_argument('--output-dir', type=str, default='./output_videos')
parser.add_argument('--input-classifier-file', type=str, default='./classifier_data.p')

# Input video range (optional)
parser.add_argument('--input-video-range', type=str, default='[]')

# Other params
parser.add_argument('--window-scales', type=str, default=default_window_scales)
parser.add_argument('--window-min-score', type=float, default=1.5)
parser.add_argument('--match-threshold', type=float, default=1.0)
parser.add_argument('--match-min-size', type=str, default='[16,16]')
parser.add_argument('--match-average-frames', type=int, default=20)
parser.add_argument('--match-average-recalc', type=int, default=100)
parser.add_argument('--video-frame-save-interval', type=int, default=40)

args = parser.parse_args()
print('Args: {}'.format(args))

# Transliterate params
input_path = args.input_path
output_dir = args.output_dir
input_classifier_file = args.input_classifier_file
input_video_range = tuple(json.loads(args.input_video_range))
window_scales = tuple(json.loads(args.window_scales))
window_min_score = args.window_min_score
match_threshold = args.match_threshold
match_min_size = tuple(json.loads(args.match_min_size))
match_average_frames = args.match_average_frames
match_average_recalc = args.match_average_recalc
video_frame_save_interval = args.video_frame_save_interval

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


def get_hit_boxes(input_image,
                  hog_size=(64, 64),
                  px_per_cell=8,
                  min_score=1.0,
                  scales=None,
                  output_base=None):
    """
    Finds & vectorizes HOG features in supplied image.
    """
    x_cells_per_window = hog_size[1] // px_per_cell - 1
    y_cells_per_window = hog_size[0] // px_per_cell - 1

    output_hits = []
    image_scale_x = image_size[0] / hog_size[0]
    image_scale_y = image_size[1] / hog_size[1]

    scale_ctr = 0
    for scale, overlap, x_range, y_range in scales:
        # Subset image
        target_x = (int(x_range[0] * input_image.shape[1]),
                    int(x_range[1] * input_image.shape[1]))
        target_y = (int(y_range[0] * input_image.shape[0]),
                    int(y_range[1] * input_image.shape[0]))
        target_part = input_image[target_y[0]:target_y[1], target_x[0]:target_x[1], :]

        # Scale subset
        target_shape = (int(target_part.shape[1] * scale), int(target_part.shape[0] * scale))
        scaled_target = cv2.resize(target_part, target_shape)

        # Spatial channels
        spatial_shape = (int(target_shape[0] * image_scale_y),
                         int(target_shape[1] * image_scale_x))
        spatial_channels = cv2.resize(scaled_target, spatial_shape)

        # Histogram channels
        histogram_channels = \
            [windowed_histogram(
                (scaled_target[:, :, histogram_channel] * 255.0 / 256.0 * histogram_bins).astype(np.uint8),
                selem=np.ones(hog_size),
                shift_x=-hog_size[1] // 2,
                shift_y=-hog_size[0] // 2,
                n_bins=histogram_bins)
                for histogram_channel in range(scaled_target.shape[-1])]

        # HOG channels
        if hog_channel == 'ALL':
            hog_channel_range = range(scaled_target.shape[-1])
        else:
            hog_channel_range = [hog_channel]

        hog_channels = []
        for hog_channel_part in hog_channel_range:
            if output_base is None:
                found_parts = \
                    classifier.get_hog_features(scaled_target[:, :, hog_channel_part],
                                                orientation=hog_orientation,
                                                px_per_cell=hog_px_per_cell,
                                                cell_per_blk=hog_cell_per_blk,
                                                feature_vector=False,
                                                visualize=False)
            else:
                (found_parts, hog_image) = \
                    classifier.get_hog_features(scaled_target[:, :, hog_channel_part],
                                                orientation=hog_orientation,
                                                px_per_cell=hog_px_per_cell,
                                                cell_per_blk=hog_cell_per_blk,
                                                feature_vector=False,
                                                visualize=True)
                image.save_image(hog_image * 100, output_base,
                                 input_color_space='GRAY',
                                 output_type='hog_{}_{}'.format(hog_channel_part, scale_ctr))

            hog_channels.append(found_parts)

        hog_shape = hog_channels[0].shape

        # X/Y ranges for walking features
        x_start = 0
        x_stop = hog_shape[1] - x_cells_per_window + 1
        x_step = int((1 - overlap) * x_cells_per_window)

        y_start = 0
        y_stop = hog_shape[0] - y_cells_per_window + 1
        y_step = int((1 - overlap) * y_cells_per_window)

        for x_px in range(x_start, x_stop, x_step):
            for y_px in range(y_start, y_stop, y_step):
                # Build cell
                window_start_x = int(x_px * px_per_cell * image_scale_x)
                window_end_x = window_start_x + image_size[0]
                window_start_y = int(y_px * px_per_cell * image_scale_y)
                window_end_y = window_start_y + image_size[1]

                # Spatial feature vector
                spatial_features = spatial_channels[window_start_y:window_end_y, window_start_x:window_end_x, :].ravel()

                # Color feature vector
                histogram_features = np.ravel(
                    [item[(y_px * px_per_cell), (x_px * px_per_cell), :].ravel()
                     for item in histogram_channels])

                # Hog feature vector
                hog_features = np.ravel(
                    [item[y_px:y_px + y_cells_per_window, x_px:x_px + x_cells_per_window].ravel()
                     for item in hog_channels])

                # Build window
                window_ul = (target_x[0] + int(x_px / scale * px_per_cell),
                             target_y[0] + int(y_px / scale * px_per_cell))
                window_lr = (int(window_ul[0] + hog_size[1] / scale),
                             int(window_ul[1] + hog_size[0] / scale))

                # Vectorize all features
                all_features = np.concatenate((spatial_features, histogram_features, hog_features))
                all_features = all_features.reshape(1, -1)
                all_features = input_scaler.transform(all_features)

                # Score & check
                classifier_score = input_classifier.decision_function(all_features)
                if classifier_score >= min_score:
                    output_hits.append((window_ul, window_lr, scale ** 2))

        scale_ctr = scale_ctr + 1

    return output_hits


def process_image(input_image,
                  mean_heatmap=None,
                  hog_size=(64, 64),
                  px_per_cell=8,
                  min_score=1.0,
                  scales=None,
                  threshold=1.0,
                  min_size=(16, 16),
                  output_base=None):
    """
    Finds and marks matching objects in supplied image.
    """
    hit_boxes = get_hit_boxes(input_image,
                              hog_size=hog_size,
                              px_per_cell=px_per_cell,
                              min_score=min_score,
                              scales=scales,
                              output_base=output_base)
    heat_map = classifier.get_heat_map(input_image.shape[0:2], hit_boxes)

    if not mean_heatmap is None:
        heat_map = mean_heatmap.update_mean(heat_map)

    binary_map = heat_map >= threshold
    labels = label_image(binary_map)

    if not output_base is None:
        image.save_image(classifier.draw_boxes(input_image, hit_boxes), output_base,
                         input_color_space=color_space, output_type='boxes')
        image.save_image(heat_map.astype(np.uint8) * 20, output_base,
                         input_color_space='GRAY', output_type='heat_map')
        image.save_image(binary_map, output_base,
                         input_color_space='GRAY', output_type='binary_map')

    # Max bound the hit boxes w/r/t the labels
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
    input_name = os.path.basename(input_file)
    input_base, input_ext = os.path.splitext(input_name)

    if input_ext in ['.jpg', '.png']:
        # Build output base filename
        output_base = os.path.join(output_dir, input_base)

        print('Loading {} (image)...'.format(input_file))
        input_image = image.load_image(input_file, color_space)
        print('...{} (image) loaded.'.format(input_file))

        print('Detecting vehicles...')
        found_boxes = process_image(input_image, mean_heatmap=None,
                                    hog_size=hog_size, px_per_cell=hog_px_per_cell,
                                    min_score=window_min_score, scales=window_scales,
                                    threshold=match_threshold, min_size=match_min_size,
                                    output_base=output_base)
        print('...Vehicles detected: {}'.format(found_boxes if len(found_boxes) > 0 else '(none)'))
        output_image = classifier.draw_boxes(image.convert_image_to_rgb(input_image, color_space),
                                             found_boxes)

        image.save_image(input_image, output_base, input_color_space=color_space, output_type='input')
        image.save_image(output_image, output_base, output_type='output')

    elif input_ext in ['.mp4']:
        # Build running mean, set frame ctr
        output_file = os.path.join(output_dir, '{}.mp4'.format(input_base))
        mean_heatmap = running_mean.RunningMean(max_size=match_average_frames,
                                                recalc_interval=match_average_recalc)
        video_frame_ctr = 0


        def process_frame(input_frame):
            global video_frame_ctr
            output_base = None

            if video_frame_save_interval > 0 and \
                    (video_frame_ctr % video_frame_save_interval) == 0:
                output_base = os.path.join(output_dir, '{}_{:0>6d}'.format(input_base, video_frame_ctr))

            work_frame = image.convert_video_colorspace(input_frame, color_space)
            found_boxes = process_image(work_frame, mean_heatmap=mean_heatmap,
                                        hog_size=hog_size, px_per_cell=hog_px_per_cell,
                                        min_score=window_min_score, scales=window_scales,
                                        threshold=match_threshold, min_size=match_min_size,
                                        output_base=output_base)
            output_frame = classifier.draw_boxes(input_frame,
                                                 found_boxes)

            if not output_base is None:
                image.save_image(input_frame, output_base, output_type='input')
                image.save_image(output_frame, output_base, output_type='output')

            video_frame_ctr = video_frame_ctr + 1
            return output_frame


        video_clip = VideoFileClip(input_file)

        if input_video_range and len(input_video_range) == 2:
            video_clip = video_clip.subclip(input_video_range[0], input_video_range[1])
        video_clip = video_clip.fl_image(process_frame)

        print('Saving {} (video)...'.format(output_file))
        video_clip.write_videofile(output_file, audio=False)
        print('...{} (video) saved.'.format(output_file))
    else:
        raise IOError('invalid file extensoin: {}' + input_ext)
