import pickle
import glob
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip

import helpers
import lanes

# Compute perspective transform matrix
persp_src = np.float32([[285, 664], [1012, 664], [685, 450], [596, 450]])
td_width = 1280
ll_targ = td_width * 7 / 16
rl_targ = td_width * 9 / 16
td_height = 1280
persp_dst = np.float32([[ll_targ, td_height-10], [rl_targ, td_height-10], [rl_targ, td_height//4], [ll_targ, td_height//4]])
M = cv2.getPerspectiveTransform(persp_src, persp_dst)

# Assumed values given 3.7m wide lanes with 3.0m long dashed lines
# pixels in perspective transformed space
ym_per_px = 3.0 / 100
xm_per_px = 3.7 / 163

#Create triangular convolution window 
window_width = 40
window = np.mgrid[:window_width // 2]
window = np.concatenate([window, window[::-1]])

def process(img, mtx, dist, db=None):
    db.s(img, "input")
    
    # Apply camera calibration
    img = cv2.undistort(img, mtx, dist, None, mtx)
    db.s(img, "undistorted")

    # Color space conversions
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    s = hls[:,:,1]
    l = hls[:,:,2]

    # Find yellow lines
    ret, yellow = cv2.threshold(h, 17, 255, cv2.THRESH_BINARY) # h > 17
    ret, yellow2 = cv2.threshold(h, 34, 255, cv2.THRESH_BINARY_INV) # h < 34
    yellows = cv2.bitwise_and(yellow, yellow2) # 17 < h < 34
    ret, l = cv2.threshold(l, 90, 255, cv2.THRESH_BINARY) # l > 90
    yellows = cv2.bitwise_and(yellows, l)
    db.s(yellows, "yellows")

    # Find white lines
    ret, whites = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY) # s > 200
    db.s(whites, "whites")

    # Combine color masks
    color_mask =  cv2.bitwise_or(yellows, whites)
    db.s(color_mask, "color_mask")

    # Perspective transform
    cw = cv2.warpPerspective(color_mask, M, dsize=(td_width, td_height), flags=cv2.INTER_LINEAR)
    db.s(cw, "cm_warp")


    l = lanes.LaneFinder(ll_targ, window_width, td_width)
    r = lanes.LaneFinder(rl_targ, window_width, td_width)

    th = cw
    lf_debug = np.dstack([th, th, th]) #np.zeros((th.shape[0], th.shape[1], 3))
    lf_debug = cv2.cvtColor(cw, cv2.COLOR_GRAY2BGR)
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
        ret, nlc = l.find_lane(conv, i)
        if ret: 
            lf_debug[len(lf_debug)-i,int(nlc)] = [0,0,255]

        ret, nrc = r.find_lane(conv, i)
        if ret: 
            lf_debug[len(lf_debug)-i,int(nrc)] = [0,0,255]
        
        # Calculate polyfit coefs
        lf = l.uf2(i)
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
    db.s(lf_debug, "lane_finder")
    # Find polynomials
    lane_overlay = np.zeros_like(lf_debug)

    for i in range(len(cw)):
        i += 1
        left = int(l.get_fit(i))
        right = int(r.get_fit(i))
        if i < max(l.ys):
            lane_overlay[len(cw) - i,left-10:left] = [0, 0, 255]
        if i < max(r.ys):
            lane_overlay[len(cw) - i,right:right+10] = [0, 0, 255]
        if i < max(l.ys) and i < max(r.ys):
            lane_overlay[len(cw) - i,left:right] = [0, 255, 0]

    
    #Calculate distance to center of lane
    if l.fit2 is not None and r.fit2 is not None:
        center = (r.fit2[-1] - l.fit2[-1]) / 2 + l.fit2[-1]
        offset = (640 - center) * xm_per_px
        cv2.putText(img, f"Center: {offset:0.3f}m", (100,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    else:
        cv2.putText(img, f"Center: Not found.", (100,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    # Calculate radius
    y_eval = 0
    if l.fit2 is not None:
        l.ys = np.array(l.ys) * ym_per_px
        l.c = np.array(l.c) * xm_per_px
        l.uf2(0)
        rad_l = np.sqrt((1 + (2 * l.fit2[0] * y_eval + l.fit2[1])**2)**3) / (2 * l.fit2[0])
        cv2.putText(img, f"Radius left : {rad_l:0.0f}m", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    else:
        cv2.putText(img, f"Radius left : Not found.", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    if r.fit2 is not None:
        r.ys = np.array(r.ys) * ym_per_px
        r.c = np.array(r.c) * xm_per_px
        r.uf2(0)
        rad_r = np.sqrt((1 + (2 * r.fit2[0] * y_eval + r.fit2[1])**2)**3) / (2 * r.fit2[0])
        cv2.putText(img, f"Radius right: {rad_r:0.0f}m", (100,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    else:
        cv2.putText(img, f"Radius right: Not found.", (100,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        

    # Overlay lane diagram
    db.s(lane_overlay, "lo")
    lane_overlay = cv2.warpPerspective(lane_overlay, M, dsize=(color_mask.shape[1], color_mask.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_NEAREST)
    img = cv2.addWeighted(img, 1, lane_overlay, 0.5, 0)

    return img

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
        break #if name != r"test_images\straight_lines1.jpg": continue
        print(name)
        db = helpers.PipelineDebug(name, "output_images")
        db.enable = True
        img = process(cv2.imread(name), cal['mtx'], cal['dist'], db=db)
        db.s(img, "final")

    for name in glob.glob("test_images/*.mp4"):
        if name != r"test_images\project_video.mp4": continue
        clip = VideoFileClip(name).subclip(0,0.5)
        bn = os.path.basename(name)
        #class mutInt:
        #    pass 
        #i = mutInt()
        #i.i = 0
        def pvid(frame):#, i=i):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            db = helpers.PipelineDebug(name, "output_images")
            #db.img_name += str(i.i)
            #db.img_ext = ".png"
            db.enable = False
            #if i.i % 100 == 0:
            #    db.enable = True
            frame = process(frame, cal['mtx'], cal['dist'], db=db)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #i.i += 1
            return frame#np.dstack([chan, chan, chan])
        xform = clip.fl_image(pvid)
        xform.write_videofile(os.path.join("output_videos", bn), audio=False)
        #break
