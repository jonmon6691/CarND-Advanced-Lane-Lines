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
    fit_len = 100

    noise_window = np.multiply(window, window)
    noise_window = window-np.mean(window)
    max_ni = sum(noise_window[noise_window > 0] * 255)

    class LaneFinder:
        def __init__(self, initial_position):
            self.c = [initial_position] # Record of detected lane centers
            self.g = [True] # Record of when the lane was detected
            self.ys = [0] # Record of the y indexes where lane was found (1 is bottom)
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

        def uf2(self, i):
            xs = np.array(self.c)[self.g]
            ys = self.ys
            self.fit2 = np.polyfit(ys, xs, 2)
            x = self.fit2[0] * i**2 + self.fit2[1] * i + self.fit2[2]
            if x!= x: # Handle NaN and inf conditions
                return xs[0]
            return np.int(np.clip(x, 0, td_width-1))

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
            # Window should move towards an offset away from other lane line
            self.f(strength * (other_lane.c[0] - self.wc + offset))

        def seek_center(self, strength):
            # Window should move toward the last lane line center found
            self.f(strength * (self.c[0] - self.wl - window_width / 2))

        def seek_fit(self, i):
            fit_location = self.fit[0] * i + self.fit[0]
            self.f(0)

        def ni(self, img_row): # Noise index, should approach zero with the confidence in the lane position
            # Super naeive impl. If you have pixels on both edges of the window, its probably bad
            if img_row[self.wl] > 128 and img_row[self.wu] > 128:
                self.norm_ni = 0
            else:
                self.norm_ni = 1
            return self.norm_ni

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

        # Find the max of the convolution within the search window, mark it
        ret, nlc = l.find_lane(conv)
        if ret: 
            conv_img[len(img)-i,int(nlc)] = [0,0,255]
            l.ys.insert(0, i)

        ret, nrc = r.find_lane(conv)
        if ret: 
            conv_img[len(img)-i,int(nrc)] = [0,0,255]
            r.ys.insert(0, i)
        
        # Calculate polyfit coefs
        l.update_fit()
        lf = l.uf2(i)
        r.update_fit()
        rf = r.uf2(i)

        conv_img[len(img)-i,int(lf)] = [0,255,255]
        conv_img[len(img)-i,int(rf)] = [0,255,255]

        # Logic to move the search window
        if l.g[0]: # If a lane was found on this row
             # Search window should seek to center it
             # Note that strength is determined by noise index,
             # if we are uncertain about the pick, 
             # then it should have less impact on the window
            l.seek_center(0.1*l.ni(row))
        else: # if you can't find anything
            l.seek_fit(i) # Move based on fit
            if r.g[0]: # And if the other lane is found
                l.seek_lane(0.1, r, -160) # track with it

        #Repeat for right lane
        if r.g[0]:
            r.seek_center(0.1*r.ni(row))
        else:
            r.f(-r.fit[0])
            if l.g[0]:
                r.seek_lane(0.01, l, 160)

        l.tick(1)
        r.tick(1)

    return np.array(conv_img)

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        if name != r"test_images\test3.jpg": continue
        print(name)
        img = process(cv2.imread(name), cal['mtx'], cal['dist'])
        cv2.imwrite(os.path.join("output_images", os.path.splitext(os.path.basename(name))[0]+".png"), img)

    for name in glob.glob("test_images/*.mp4"):
        #break
        clip = VideoFileClip(name)
        bn = os.path.basename(name)
        def pvid(i):
            chan = process(i, cal['mtx'], cal['dist'])
            return np.dstack([chan, chan, chan])
        xform = clip.fl_image(pvid)
        xform.write_videofile(os.path.join("output_videos", bn), audio=False)
        #break
