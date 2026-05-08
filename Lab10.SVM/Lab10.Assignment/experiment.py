import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/project_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # Extract HOG, color histogram, and spatial features from sample images and
    # return results and parameters in a dict.
    feature_data = processFiles(
        pos_dir, neg_dir, recurse=True,
        color_space="ycrcb", channels=[0, 1, 2],
        hog_features=True, hist_features=True, spatial_features=True,
        hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2),
        hist_bins=32, spatial_size=(16, 16))


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data, C=0.1)


    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector(init_size=(64, 64), x_overlap=0.75, y_step=0.04,
                        x_range=(0.05, 0.95), y_range=(0.55, 1.0),
                        scale=1.2).loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap, num_frames=12, threshold=90,
                         show_video=False, draw_heatmap=True, write=True,
                         write_fps=24, output_filename="result_video.mp4")


# def experiment2
#    ...

if __name__ == "__main__":
    experiment1()
    # experiment2() may you need to try other parameters
    # experiment3 ...


