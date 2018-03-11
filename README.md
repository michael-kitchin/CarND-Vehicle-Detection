**Vehicle Detection Project**

## Writeup

### Introduction

The goals / steps of this project are the following:

* Perform Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images to train a Linear SVM classifier.
* Use a trained classifier coupled with a sliding-window technique to search for vehicles in images/videos.
* Run this pipeline on a video stream to create heat maps (integrations) of recurring detections for outlier rejection and vehicle following.
* Estimate and render bounding boxes for vehicles detected.

### Environment

Minimum execution environment is undefined.

Project was developed using the following environment:

| Category | Item        |
|----------|-------------|
| OS       | Windows 10 |
| CPU      | Intel i7/6800k |
| RAM      | 64GB |
| GPU      | nVidia GTX 1060 |
| VRAM     | 6GB |
| Storage  | SATA SSD |

[//]: # (Image References)

[this]: https://github.com/michael-kitchin/CarND-Vehicle-Detection
[train_classifier.py]: ./train_classifier.py
[process.py]: ./process.py
[image.py]: ./util/image.py
[classifier.py]: ./util/classifier.py
[running_mean.py]: ./util/running_mean.py

### Execution

This capability consists of the following scripts:
* [train_classifier.py] Creates and trains a scaler/classifier on a supplied collection of car/non-car images, then stores these objects with their configuration in a Python pickle file for subsequent execution.
* [process.py] Loads previously trained scaler/classifier and their configuration, then applies these to supplied images and/or videos in order to detect and highlight cars.
* [image.py] Library of image management functions based on course material.
* [classifier.py] Library of classifier-related functions based on course material.
* [running_mean.py] Support class for managing running means of NumPy arrays.

#### train_classifier.py

Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value |
|:-------:|-------------|----------------------|
| `--vehicle-image-path` | Vehicle training image path. | `./training_images/vehicles` |
| `--non-vehicle-image-path` |  Non-vehicle training image path. | `./training_images/non-vehicles` |
| `--output-classifier-file` | File name for stored, trained objects. | `./classifier_data.p` |
| `--test-fraction` | Fraction of training images to set aside for verification. | `0.2` |
| `--color-space` | Color space for training/execution. | `YCrCb` |
| `--image-size` | Image size to scale training/execution images to (in pixels). | `[32,32]` |
| `--histogram-bins` | Number of histogram bins for color feature detection. | `32` |
| `--hog-size` | HOG window size (in pixels). | `[64,64]` |
| `--hog-orientation` | HOG orientation count. | `9` |
| `--hog-px-per-cell` | HOG pixels per cell. | `8` |
| `--hog-cell-per-blk` | HOG cells per block. | `2` |
| `--hog-channel` | HOG channel | `ALL`, `0`,`1`, or`2` |

Example execution (Windows, w/in JetBrains IntelliJ):
```
C:\Tools\Anaconda3\envs\bc-project-gpu-1\python.exe C:\Users\mcoyo\.IntelliJIdea2017.3\config\plugins\python\helpers\pydev\pydev_run_in_console.py 55719 55720 E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection/train_classifier.py --vehicle-image-path=./training_images/vehicles --non-vehicle-image-path=./training_images/non-vehicles
Running E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection/train_classifier.py
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['E:\\Projects\\Work\\Learning\\CarND\\CarND-Vehicle-Detection', 'E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection'])
Python 3.5.4 | packaged by conda-forge | (default, Dec 18 2017, 06:53:03) [MSC v.1900 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
Args: Namespace(color_space='YCrCb', histogram_bins=32, hog_cell_per_blk=2, hog_channel='ALL', hog_orientation=9, hog_px_per_cell=8, hog_size='[64,64]', image_size='[32,32]', non_vehicle_image_path='./training_images/non-vehicles', output_classifier_file='./classifier_data.p', test_fraction=0.2, vehicle_image_path='./training_images/vehicles')
Extracting vehicle features...
8792 vehicle features extracted.
Extracting non-vehicle features...
8968 non-vehicle features extracted.
Training classifier...
Classifier trained.
 True/Positive: 99.2%
False/Positive: 0.3%
 True/Negative: 99.7%
False/Negative: 0.8%
Saving classifier to ./classifier_data.p.
Classifier saved to ./classifier_data.p.
```

#### process.py

Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value |
|:-------:|-------------|----------------------|
| `--input-path` | Input search path for images/videos (supports wildcards and MP4, JPG, and PNG files). | `./*.mp4` |
| `--output-dir` | Output directory for processed images/videos and interim images. | `./output_videos` |
| `--input-video-range` | Start/stop subset of input video to process (in seconds). | `[]`=full video, `[x,y]`=(x) to (y) |
| `--window-scales` | Scales and image fractions for HOG feature extraction. | (see below) |
| `--window-min-score` | Minimum decision function score for inclusion.| `1.5` |
| `--match-threshold` | Minimum averaged heatmap score for inclusion. | `1.0` |
| `--match-min-size` | Minimum detected feature size for inclusion (in pixels). | `[16,16]` |
| `--match-average-frames` | Number of frame heatmaps to average. | `20` |
| `--match-average-recalc` | Frame interval to recalc average heatmap. | `100` |
| `--video-frame-save-interval` | Frame interval to save interim images. | `40` |

Example execution (Windows, w/in JetBrains IntelliJ):
```
C:\Tools\Anaconda3\envs\bc-project-gpu-1\python.exe C:\Users\mcoyo\.IntelliJIdea2017.3\config\plugins\python\helpers\pydev\pydev_run_in_console.py 56761 56762 E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection/process.py --input-path=./test_images/*.jpg --output-dir=./output_images --input-video-range=[0.0,2.0]
Running E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection/process.py
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['E:\\Projects\\Work\\Learning\\CarND\\CarND-Vehicle-Detection', 'E:/Projects/Work/Learning/CarND/CarND-Vehicle-Detection'])
Python 3.5.4 | packaged by conda-forge | (default, Dec 18 2017, 06:53:03) [MSC v.1900 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
Args: Namespace(input_classifier_file='./classifier_data.p', input_path='./test_images/*.jpg', input_video_range='[0.0,2.0]', match_average_frames=20, match_average_recalc=100, match_min_size='[16,16]', match_threshold=1.0, output_dir='./output_images', video_frame_save_interval=40, window_min_score=1.5, window_scales='[\n    [0.5, 0.75, [0.0, 1.0], [0.5, 0.9]],\n    [0.6, 0.75, [0.0, 1.0], [0.5, 0.9]],\n    [1.0, 0.5, [0.3333, 0.6666], [0.55, 0.9]],\n    [1.3333, 0.5, [0.0, 1.0], [0.5, 0.75]],\n    [2.0, 0.0, [0.25, 0.75], [0.55, 0.65]]\n]')
Classifier args: Namespace(color_space='YCrCb', histogram_bins=32, hog_cell_per_blk=2, hog_channel='ALL', hog_orientation=9, hog_px_per_cell=8, hog_size='[64,64]', image_size='[32,32]', non_vehicle_image_path='./training_images/non-vehicles', output_classifier_file='./classifier_data.p', test_fraction=0.2, vehicle_image_path='./training_images/vehicles')
Loading ./test_images\test1.jpg (image)...
./test_images\test1.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: [((800, 373), (959, 519)), ((1040, 373), (1278, 519))]
Loading ./test_images\test2.jpg (image)...
./test_images\test2.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: (none)
Loading ./test_images\test3.jpg (image)...
./test_images\test3.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: [((900, 414), (947, 461))]
Loading ./test_images\test4.jpg (image)...
./test_images\test4.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: [((800, 376), (975, 519)), ((1040, 376), (1265, 535))]
Loading ./test_images\test5.jpg (image)...
./test_images\test5.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: [((800, 360), (975, 519)), ((1080, 392), (1231, 519))]
Loading ./test_images\test6.jpg (image)...
./test_images\test6.jpg (image) loaded.
Detecting vehicles...
Vehicles detected: [((800, 360), (959, 519)), ((1000, 376), (1215, 535))]
```
---

## Rubric Points

### [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) are discussed individually with respect to the implementation.

---

### 1. Writeup / README

#### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one (...).

_The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled._

See: GitHub [repo][this].

---

### 2. Histogram of Oriented Gradients (HOG)
       
#### 2.1 Explain how (and identify where in your code) you extracted HOG features from the training images (...).

_Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why._

#### 2.2 Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

_The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier._

---

