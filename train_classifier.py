import argparse
import json
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from util import classifier, image

parser = argparse.ArgumentParser()

# Input files
parser.add_argument('--vehicle-image-path', type=str, default='./training_images/vehicles')
parser.add_argument('--non-vehicle-image-path', type=str, default='./training_images/non-vehicles')
parser.add_argument('--output-classifier-file', type=str, default='./classifier_data.p')

# Other parameters
parser.add_argument('--color-space', type=str, default='YCrCb')
parser.add_argument('--image-size', type=str, default='[32, 32]')
parser.add_argument('--histogram-bins', type=int, default=32)
parser.add_argument('--hog-size', type=str, default='[64, 64]')
parser.add_argument('--hog-orientation', type=int, default=9)
parser.add_argument('--hog-px-per-cell', type=int, default=8)
parser.add_argument('--hog-cell-per-blk', type=int, default=2)
parser.add_argument('--hog-channel', type=str, default='ALL')

args = parser.parse_args()
print('Args: {}'.format(args))

# Transliterate params
vehicle_image_path = args.vehicle_image_path
non_vehicle_image_path = args.non_vehicle_image_path
output_classifier_file = args.output_classifier_file
color_space = args.color_space
image_size = tuple(json.loads(args.image_size))
hog_size = tuple(json.loads(args.hog_size))
histogram_bins = args.histogram_bins
hog_orientation = args.hog_orientation
hog_px_per_cell = args.hog_px_per_cell
hog_cell_per_blk = args.hog_cell_per_blk

# May be 'ALL', 'NONE', or integer
try:
    hog_channel = int(args.hog_channel)
except ValueError:
    hog_channel = args.hog_channel

print('Extracting vehicle features...')
vehicle_features = classifier.extract_features_from_images(
    image.find_image_files(vehicle_image_path),
    color_space=color_space,
    image_size=image_size,
    histogram_bins=histogram_bins,
    hog_orientation=hog_orientation,
    hog_px_per_cell=hog_px_per_cell,
    hog_cell_per_blk=hog_cell_per_blk,
    hog_channel=hog_channel)
print('...{} vehicle features extracted.'.format(len(vehicle_features)))

print('Extracting non-vehicle features...')
non_vehicle_features = classifier.extract_features_from_images(
    image.find_image_files(non_vehicle_image_path),
    color_space=color_space,
    image_size=image_size,
    histogram_bins=histogram_bins,
    hog_orientation=hog_orientation,
    hog_px_per_cell=hog_px_per_cell,
    hog_cell_per_blk=hog_cell_per_blk,
    hog_channel=hog_channel)
print('...{} non-vehicle features extracted.'.format(len(non_vehicle_features)))

X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

output_scaler = StandardScaler().fit(X_train)
X_train = output_scaler.transform(X_train)
X_test = output_scaler.transform(X_test)

print('Training classifier...')
output_classifier = LinearSVC()
output_classifier.fit(X_train, y_train)
print('...Classifier trained.')

pos_results = y_test.nonzero()
neg_results = np.logical_not(y_test).nonzero()
true_pos_score = output_classifier.score(X_test[pos_results], y_test[pos_results])
true_neg_score = output_classifier.score(X_test[neg_results], y_test[neg_results])
print(' True/Positive: {:.1f}%'.format(100.0 * true_pos_score))
print('False/Positive: {:.1f}%'.format(100.0 * (1.0 - true_neg_score)))
print(' True/Negative: {:.1f}%'.format(100.0 * true_neg_score))
print('False/Negative: {:.1f}%'.format(100.0 * (1.0 - true_pos_score)))

print('Saving classifier to {}.'.format(output_classifier_file))
with open(output_classifier_file, 'wb') as classifier_file:
    data = {
        'args': args,
        'scaler': output_scaler,
        'classifier': output_classifier
    }
    pickle.dump(data, classifier_file)
print('...Classifier saved to {}.'.format(output_classifier_file))
