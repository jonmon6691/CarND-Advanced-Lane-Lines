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

#Create triangular convolution window 
window_width = 20
window = np.mgrid[:window_width // 2]
window = np.concatenate([window, window[::-1]])

def process(img, mtx, dist):
    # Apply camera calibration
    img = cv2.undistort(img, mtx, dist, None, mtx)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #h = hls[:,:,0]
    #s = hls[:,:,1]
    l = hls[:,:,2]

    dx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    dx = np.uint8(255*dx/np.max(dx))

    img = np.zeros_like(gray)
    img[l > 128] = 255
    img[dx > 30] = 255

    # Apply perspective transform
    img = cv2.warpPerspective(img, M, dsize=(td_width, td_height), flags=cv2.INTER_LINEAR)



    conv_img = np.dstack([img, img, img])
    fit_len = 200
    class LaneFinder:
        def __init__(self, initial_position):
            self.c = [initial_position] # Record of detected lane centers
            self.g = [True] # Record of when the lane was detected
            self.wc = self.c[0] # curent search window center
            self.dwc = 0 # current dy/dt of search window center

            self.fit = [0,0] # last fit line coef.s

        @property
        def wl(self): # current window lower bound
            return int(np.clip(self.wc - window_width / 2, 0, td_width - window_width - 1))

        @property
        def wu(self): # current window upper bound
            return int(np.clip(self.wc + window_width / 2, window_width, td_width - 1))

        def update_fit(self):
            mask = self.g[:fit_len]
            if sum(mask) > fit_len/2:
                xs = np.array(self.c[:fit_len])
                ys = np.mgrid[:len(xs)]
                self.fit = np.polyfit(ys[mask], xs[mask], 1)

        def find_lane(self, conv):
            self.winconv = conv[self.wl: self.wu]
            if np.max(self.winconv) > 10:
                nlc = np.clip(np.argmax(self.winconv) + self.wl, 0, td_width-1)
                self.g.insert(0, True)
            else:
                nlc = self.c[0]
                self.g.insert(0, False)
            self.c.insert(0, nlc)
            return self.g[0], nlc

        def f(self, f): # Apply force
            self.dwc += f

        def tick(self, dt): # Integrate applied force, reset force accumulator
            self.wc += self.dwc * dt
            self.dwc = 0

        def seek_lane(self, strength, other_lane, offset):
            self.f(strength * (other_lane.c[0] - self.wc + offset))

        def seek_center(self, strength):
            self.f(strength * (self.c[0] - self.wl - window_width / 2))

    l = LaneFinder(ll_targ)
    r = LaneFinder(rl_targ)
    
    for i, row in enumerate(img[::-1], start=1):
        # Draw seach windows
        conv_img[len(img)-i,l.wl] = [255,255,0]
        conv_img[len(img)-i,l.wu] = [255,255,0]

        conv_img[len(img)-i,r.wl] = [255,255,0]
        conv_img[len(img)-i,r.wu] = [255,255,0]
        
        # Convol window with row of pixels, crop the overhang
        conv = np.convolve(window, row)[window_width // 2 - 1:1-window_width // 2]

        # Find the max of the convolution within the search window
        ret, nlc = l.find_lane(conv)
        #noise_indicator = np.sum(np.multiply(window-, l.winconv))
        if ret: conv_img[len(img)-i,int(nlc)] = [0,0,255]
        ret, nrc = r.find_lane(conv)
        if ret: conv_img[len(img)-i,int(nrc)] = [0,0,255]
        
        # Calculate polyfit coefs
        l.update_fit()
        r.update_fit()


        # Logic to move the search window
        l.f(-l.fit[0]) # Move based on fit
        if l.g[0]: # If a lane was found
            l.seek_center(0.1) # Search window should seek to center it
        else: # if you can't find anything
            if r.g[0]: # But the other lane can
                l.seek_lane(0.1, r, -160) # track with the other lane

        r.f(-r.fit[0])
        if r.g[0]:
            r.seek_center(0.1)
        else: # if you can't find anything
            if r.g[0]: # But the other lane can
                r.seek_lane(0.1, l, 160) # track with the other lane

        l.tick(1)
        r.tick(1)

    return np.array(conv_img)

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        #if name != r"test_images\test3.jpg": continue
        print(name)
        img = process(cv2.imread(name), cal['mtx'], cal['dist'])
        cv2.imwrite(os.path.join("output_images", os.path.splitext(os.path.basename(name))[0]+".png"), img)

    for name in glob.glob("*.mp4"):
        break
        clip = VideoFileClip(name)
        def pvid(i):
            chan = process(i, cal['mtx'], cal['dist'])
            return np.dstack([chan, chan, chan])
        xform = clip.fl_image(pvid)
        xform.write_videofile(os.path.join("output_videos", name), audio=False)
        break
