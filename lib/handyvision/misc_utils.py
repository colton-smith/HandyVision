""" misc_utils.py

Misc. utils.
"""
import time 

class FPSCounter:
    def __init__(self):
        self.last_time = time.time()
        self.fps = 0
        self.frametime_s = 0 
        self.frametime_ms = 0
    
    def update(self):
        self.frametime_s = time.time() - self.last_time
        self.last_time = time.time()
        self.frametime_ms = self.frametime_s / 1000.0
        self.fps = 1.0 / max(0.00000001, self.frametime_s)
        return self.fps, self.frametime_ms
