import pickle
import glob
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip


persp_src = np.float32([[285, 664], [1012, 664], [685, 450], [596, 450]])
# TODO: The destination of the xform doesn't have to be the same dims as the img, this can actually cause loss of detail near the bottom
persp_dst = np.float32([[320, 720], [960, 720], [960, 0], [320, 0]])
M = cv2.getPerspectiveTransform(persp_src, persp_dst)

def process(img, mtx, dist):
    # Apply camera calibration
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # Apply perspective transform
    img = cv2.warpPerspective(img, M, dsize=img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        img = process(cv2.imread(name), cal['mtx'], cal['dist'])
        cv2.imwrite(os.path.join("output_images", os.path.basename(name)), img)

    for name in glob.glob("*.mp4"):
        clip = VideoFileClip(name)
        xform = clip.fl_image(lambda x:process(x, cal['mtx'], cal['dist']))
        xform.write_videofile(os.path.join("output_videos", name), audio=False)
