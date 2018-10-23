import pickle
import glob
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip


persp_src = np.float32([[285, 664], [1012, 664], [685, 450], [596, 450]])
# TODO: The destination of the xform doesn't have to be the same dims as the img, this can actually cause loss of detail near the bottom
td_width = 1280
ll_targ = td_width * 7 / 16
rl_targ = td_width * 9 / 16
td_height = 1280
persp_dst = np.float32([[ll_targ, td_height], [rl_targ, td_height], [rl_targ, td_height-320], [ll_targ, td_height-320]])
M = cv2.getPerspectiveTransform(persp_src, persp_dst)

def process(img, mtx, dist):
    # Apply camera calibration
    img = cv2.undistort(img, mtx, dist, None, mtx)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    s = hls[:,:,1]
    l = hls[:,:,2]

    dx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    dx = np.uint8(255*dx/np.max(dx))

    img = np.zeros_like(gray)
    img[l > 128] = 255
    img[dx > 30] = 255

    # Apply perspective transform
    img = cv2.warpPerspective(img, M, dsize=(td_width, td_height), flags=cv2.INTER_LINEAR)


    return img

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        img = process(cv2.imread(name), cal['mtx'], cal['dist'])
        cv2.imwrite(os.path.join("output_images", os.path.basename(name)), img)

    for name in glob.glob("*.mp4"):
        break
        clip = VideoFileClip(name)
        def pvid(i):
            chan = process(i, cal['mtx'], cal['dist'])
            return np.dstack([chan, chan, chan])
        xform = clip.fl_image(pvid)
        xform.write_videofile(os.path.join("output_videos", name), audio=False)
        break
