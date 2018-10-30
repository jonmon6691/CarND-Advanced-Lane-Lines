import numpy as np

# Static config
fit_len = 200 # Minimum number of found points before trying to fit a predictive curve
timeout = 1000 # Number of rows without a found lane to stop searching

class LaneFinder:
    def __init__(self, initial_position, window_width, img_width):
        self.ww = window_width
        self.iw = img_width

        self.c = [initial_position] # Record of detected lane centers
        self.g = [True] # Record of when the lane was detected
        self.ys = [0] # Record of the y indexes where lane was found (1 is bottom)
        self.wc = self.c[0] # curent search window center
        self.dwc = 0 # current dy/dt of search window center

        self.fit = [0,0] # last fit line coef.s

    @property
    def wl(self): # current window lower bound
        return int(np.clip(self.wc - self.ww / 2, 0, self.iw - self.ww - 1))

    @property
    def wu(self): # current window upper bound
        return int(np.clip(self.wc + self.ww / 2, self.ww, self.iw - 1))

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
            return np.int(np.clip(x, 0, self.iw-1))

    def timed_out(self):
        # Stop searching for a line if you lost it
        return len(self.g) > timeout and sum(self.g[:timeout]) == 0

    def find_lane(self, conv, i):
        self.winconv = conv[self.wl: self.wu]
        if np.max(self.winconv) > 10 and not self.timed_out():
            nlc = np.clip(np.argmax(self.winconv) + self.wl, 0, self.iw-1)
            self.g.insert(0, True)
            self.ys.insert(0, i)
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
        self.f(strength * (self.c[0] - self.wl - self.ww / 2))

    def seek_fit(self, strength, i):
        self.f(strength * (self.get_fit(i) - self.wc))

    def ni(self, img_row): # Noise index, should approach zero with the confidence in the lane position
        # Super naeive impl. If you have pixels on both edges of the window, its probably bad
        if img_row[self.wl] > 128 and img_row[self.wu] > 128:
            self.norm_ni = 0
        else:
            self.norm_ni = 1
        return self.norm_ni
