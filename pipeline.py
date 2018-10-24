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
window_width = 40
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
    lc = [ll_targ]
    lg = [True]# Stores the quality of the match
    wcl = lc[0]
    rc = [rl_targ]
    rg = [True]# Stores the quality of the match
    wcr = rc[0]

    lfit = [0, 0]
    rfit = [0, 0]
    for i, row in enumerate(img[::-1], start=1):
        conv = np.convolve(window, row)[window_width // 2 - 1:1-window_width // 2]
        
        #Move the window based on the last 20 centers
        fit_len = 200
        mask = lg[:fit_len]
        if sum(mask) > fit_len/2:
            xs = np.array(lc[:fit_len])
            ys = np.mgrid[:len(xs)]
            lfit = np.polyfit(ys[mask], xs[mask], 1)
            #wcl = lfit[0] * -1 + lfit[1]



        mask = rg[:fit_len]
        if sum(mask) > fit_len/2:
            xs = np.array(rc[:fit_len])
            ys = np.mgrid[:len(xs)]
            rfit = np.polyfit(ys[mask], xs[mask], 1)
            #wcr = rfit[0] * -1 + rfit[1]
        


        #Convol the row with the window, draw the window
        wll = int(np.clip(wcl - window_width // 2, 0, td_width-1-window_width))
        wul = int(np.clip(wcl + window_width // 2, window_width, td_width-1))
        lwin = conv[wll: wul]
        conv_img[len(img)-i,wll] = [255,255,0]
        conv_img[len(img)-i,wul] = [255,255,0]
        
        wlr = int(np.clip(wcr - window_width // 2, 0, td_width-1-window_width))
        wur = int(np.clip(wcr + window_width // 2, window_width, td_width-1))
        rwin = conv[wlr: wur]
        conv_img[len(img)-i,wlr] = [255,255,0]
        conv_img[len(img)-i,wur] = [255,255,0]
        

        try:
            if np.max(lwin) > 1:
                nlc = np.clip(np.argmax(lwin) + wll, 0, td_width-1)
                conv_img[len(img)-i,int(nlc)] = [0,0,255]
                lg.insert(0, True)
            else:
                nlc = lc[0]
                lg.insert(0, False)
            
            if np.max(rwin) > 1:
                nrc = np.clip(np.argmax(rwin) + wlr, 0, td_width-1)
                conv_img[len(img)-i,int(nrc)] = [0,0,255]
                rg.insert(0, True)
            else:
                nrc = rc[0]
                rg.insert(0, False)
        except ValueError:
            pass

        wcl -= lfit[0]
        if lg[0]:
            wcl += 0.1*(lc[0] - wll - 20)
        else:
            if rg[0]:
                wcl += 0.1*(rc[0] - wcl - 160)

        wcr -= rfit[0]
        if rg[0]:
            wcr += 0.1*(rc[0] - wlr - 20)
        else:
            if lg[0]:
                wcr += 0.1*(lc[0] - wcr + 160)

        lc.insert(0, nlc)
        rc.insert(0, nrc)
        #c = np.uint8(255*conv/np.max(conv))[:1280]
        #conv_img[len(img)-i-1] = np.dstack([c, c, c])

    return np.array(conv_img)

if __name__ == "__main__":
    with open("camera.cal", "rb") as f:
        cal = pickle.load(f)

    for name in glob.glob("test_images/*.jpg"):
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
