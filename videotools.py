#!/usr/bin/env python
import skvideo.io
import os,sys
from operator import add
import numpy as np
import boto3
import botocore
import tempfile
from pathlib import Path
import PIL
import s3tools

class Video:
    isopen=False
    _nframes=None
    _fps=None
    _dur=None
    _pth=None
    def __init__(self,pth):
        self._pth=s3tools.getpath(pth)

    def get_dur(self):
        if self._dur is None:
            # opencv metadata does not seem reliable for these files
            # ffprobe is good for duration
            metadata = skvideo.io.ffprobe(self._pth)
            self._dur=float(metadata['video']['@duration'])
        return self._dur
        
    def get_nframes(self):
        if self._nframes is None:
            metadata = skvideo.io.ffprobe(self._pth)
            self._nframes=float(metadata['video']['@nb_frames'])
        return self._nframes

    def get_fps(self):
        if self._fps is None:
            self._fps=self.get_nframes()/self.get_dur()
        return self._fps


    def open(self):
        # here you can set keys and values for parameters in ffmpeg
        self.cap= skvideo.io.FFmpegReader(self._pth)
        self.currframe=0
        self.currtime=0
        self.isopen=True
        
    def average_chunk(self,dur,outfn=None):
        allimg=np.asarray(0.)
        nframes=0
        while (self.currtime+dur)>(self.currframe/self._fps):
            if self.currframe<self.nframes:
                img = self.cap.next()
                self.currframe+=1
                allimg=np.add(allimg,img)
                nframes += 1
            else:
                self.isopen=False
                break

        # Get average
        meanimg= np.divide(allimg,nframes)

        self.currtime=self.currtime+dur

        # Write out if requested to
        if not outfn is None:
            skvideo.io.vwrite(outfn, meanimg)

        return meanimg


    def select_frames(self, dur, outfn=None):
        '''
        Picks a single frame and skips frames for "dur" from it
        '''
        nframes = 0
        firstimg=None
        while (self.currtime + dur) > (self.currframe / self._fps):
            if self.currframe<self._nframes:
                img = self.cap.next()
                if nframes==0:
                    firstimg={'currtime': self.currtime, 'currframe': self.currframe, 'img': img}
            else:
                self.isopen = False
            self.currframe += 1
            nframes += 1

        # Get average

        self.currtime = self.currtime + dur

        # Write out if requested to
        if not outfn is None and not firstimg is None:
            skvideo.io.vwrite(outfn, firstimg['img'])

        return firstimg

    def get_next_frame(self, outfn=None,skipcurrtimeupdate=False):
        try:
            img = next(self.cap.nextFrame())
            if not isinstance(img,PIL.Image.Image):
                img=PIL.Image.fromarray(img)
        except:
            self.isopen=False
            img=None


        self.currframe += 1

# Get average

        if skipcurrtimeupdate:
            self.currtime=None
        else:
            self.currtime = self.currtime + 1/self.get_fps()

        # Write out if requested to
        #        if not outfn is None:
        #            cv2.imwrite(outfn, img)

        return img






        