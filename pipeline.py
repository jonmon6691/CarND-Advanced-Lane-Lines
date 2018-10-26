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
persp_dst = np.float32([[ll_targ, td_height], [rl_targ, td_height], [rl_targ, td_height//2], [ll_targ, td_height//2]])
M = cv2.getPerspectiveTransform(persp_src, persp_dst)
Minv = cv2.getPerspectiveTransform(persp_dst, persp_src)

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

    th = np.zeros_like(gray) # thresholding
    th[l > 128] = 255
    th[dx > 30] = 255

    # Apply perspective transform
    th = cv2.warpPerspective(th, M, dsize=(td_width, td_height), flags=cv2.INTER_LINEAR)

    lf_debug = np.zeros((th.shape[1], th.shape[0], 3))
    fit_len = 200

    timeout = 200

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
            if len(xs) > 20:
                self.fit2 = np.polyfit(ys, xs, 2)
            else:
                self.fit2 = None
            return self.get_fit(i)

        def get_fit(self, i):
            if self.fit2 is None:
                return self.c[0]
            else:
                x = self.fit2[0] * i**2 + self.fit2[1] * i + self.fit2[2]
                return np.int(np.clip(x, 0, td_width-1))

        def timed_out(self):
            # Stop searching for a line if you lost it
            return len(self.g) > timeout and sum(self.g[:timeout]) == 0

        def find_lane(self, conv):
            self.winconv = conv[self.wl: self.wu]
            if np.max(self.winconv) > 10 and not self.timed_out():
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

        def seek_fit(self, strength, i):
            self.f(strength * (self.get_fit(i) - self.wc))

        def ni(self, img_row): # Noise index, should approach zero with the confidence in the lane position
            # Super naeive impl. If you have pixels on both edges of the window, its probably bad
            if img_row[self.wl] > 128 and img_row[self.wu] > 128:
                self.norm_ni = 0
            else:
                self.norm_ni = 1
            return self.norm_ni

    l = LaneFinder(ll_targ)
    r = LaneFinder(rl_targ)

    for i, row in enumerate(th[::-1], start=1): # Go line by line through the image bottom up
        # Draw search windows
        if not l.timed_out():
            lf_debug[len(lf_debug)-i,l.wl] = [255,255,0]
            lf_debug[len(lf_debug)-i,l.wu] = [255,255,0]
        if not r.timed_out():
            lf_debug[len(lf_debug)-i,r.wl] = [255,255,0]
            lf_debug[len(lf_debug)-i,r.wu] = [255,255,0]
        
        # Convol window with row of pixels, crop the overhang
        conv = np.convolve(window, row)[window_width // 2 - 1:1-window_width // 2]

        # Find the max of the convolution within the search window, mark it
        ret, nlc = l.find_lane(conv)
        if ret: 
            lf_debug[len(lf_debug)-i,int(nlc)] = [0,0,255]
            l.ys.insert(0, i)

        ret, nrc = r.find_lane(conv)
        if ret: 
            lf_debug[len(lf_debug)-i,int(nrc)] = [0,0,255]
            r.ys.insert(0, i)
        
        # Calculate polyfit coefs
        l.update_fit()
        lf = l.uf2(i)
        r.update_fit()
        rf = r.uf2(i)

        lf_debug[len(lf_debug)-i,int(lf)] = [0,255,255]
        lf_debug[len(lf_debug)-i,int(rf)] = [0,255,255]

        # Logic to move the search window
        if l.g[0]: # If a lane was found on this row
             # Search window should seek to center it
             # Note that strength is determined by noise index,
             # if we are uncertain about the pick, 
             # then it should have less impact on the window
            l.seek_center(0.051*l.ni(row))
        else: # if you can't find anything
            if r.g[0]: # And if the other lane is found
                l.seek_lane(0.025, r, -160) # track with it
            else:
                l.seek_fit(0.05, i) # Move based on fit
        #Repeat for right lane
        if r.g[0]:
            r.seek_center(0.1*r.ni(row))
        else:
            if l.g[0]:
                r.seek_lane(0.01, l, 160)
            else:
                r.seek_fit(0.1, i)

        l.tick(1)
        r.tick(1)

    # Inverse xform
    lf_debug = np.uint8(lf_debug)

    

    lf_debug = cv2.warpPerspective(lf_debug, M, dsize=(gray.shape[1], gray.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_NEAREST)
    
    mask = cv2.cvtColor(lf_debug, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.addWeighted(img, 1, lf_debug, 1, 0)

    return img

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        #if name != r"test_images\test3.jpg": continue
        print(name)
        img = process(cv2.imread(name), cal['mtx'], cal['dist'])
        cv2.imwrite(os.path.join("output_images", os.path.splitext(os.path.basename(name))[0]+".png"), img)

    for name in glob.glob("test_images/*.mp4"):
        break
        clip = VideoFileClip(name)
        bn = os.path.basename(name)
        def pvid(i):
            chan = process(i, cal['mtx'], cal['dist'])
            return chan#np.dstack([chan, chan, chan])
        xform = clip.fl_image(pvid)
        xform.write_videofile(os.path.join("output_videos", bn), audio=False)
        #break
