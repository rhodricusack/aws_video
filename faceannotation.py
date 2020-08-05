import boto3
from PIL import Image, ImageDraw
import json
import os

def markfaces(faces,im,cols):
    # Annotate image
    draw = ImageDraw.Draw(im)
    # Mark each face
    for ind,fd in enumerate([face['Face'] for face in faces]):
        # Colour-per-face specified?
        if not cols is None:
            linecol = cols[ind]

        bb = fd['BoundingBox'];
        x1 = bb['Left'] * im.size[0];
        x2 = (bb['Left'] + bb['Width']) * im.size[0]
        y1 = bb['Top'] * im.size[1];
        y2 = (bb['Top'] + bb['Height']) * im.size[1]
        draw.line((x1, y1, x2, y1), fill=linecol)
        draw.line((x2, y1, x2, y2), fill=linecol)
        draw.line((x2, y2, x1, y2), fill=linecol)
        draw.line((x1, y2, x1, y1), fill=linecol)
    return im

def marklandmarks(faces,im):
    # Annotate image
    draw = ImageDraw.Draw(im)

    for fd in [face['Face'] for face in faces]:
        for landmark in fd['Landmarks']:
            x=landmark['X']*im.size[0];
            y=landmark['Y']*im.size[1];
            draw.point((x, y),fill=(255, 255, 0, 192))

    return im


def markeyesclosed(faces,im):
    # Annotate image
    draw = ImageDraw.Draw(im)

    for fd in [face['Face'] for face in faces]:
        if not fd['EyesOpen']['Value'] :
            bb = fd['BoundingBox']
            x1 = bb['Left'] * im.size[0]
            x2 = (bb['Left'] + bb['Width']) * im.size[0]
            y1 = bb['Top'] * im.size[1]
            y2 = (bb['Top'] + bb['Height']) * im.size[1]
            gap=(y2-y1)*0.03
            crosssize=(y2-y1)*0.07
            draw.ellipse((x1+gap, y1+gap, x1+gap+crosssize, y1+gap+crosssize),fill=(128,0,0,128))
    return im

def markmanual(im,colmean):
    draw = ImageDraw.Draw(im)
    x1 = 0
    x2 = im.size[0] - 1
    y1 = 0
    y2 = im.size[1] - 1
    draw.line((x1, y1, x2, y1), fill=colmean)
    draw.line((x2, y1, x2, y2), fill=colmean)
    draw.line((x2, y2, x1, y2), fill=colmean)
    draw.line((x1, y2, x1, y1), fill=colmean)
    return(im)


def geteye(eye,fd,im,markpupil=False):
    if not eye in ['left','right']:
        raise 'Specify left or right eye'

    if im is None:
        im = Image.open(self.fn)

    if fd is None:
        if len(self.response['Face'])>1:
            raise('More than one face found in image and which one not specified')
        else:
            fd=self.response['Face'][0]

    # Find left eye
    el=next(obj for obj in fd['Landmarks'] if obj['Type']==eye+'EyeLeft')
    er=next(obj for obj in fd['Landmarks'] if obj['Type']==eye+'EyeRight')
    eu=next(obj for obj in fd['Landmarks'] if obj['Type']==eye+'EyeUp')
    ed=next(obj for obj in fd['Landmarks'] if obj['Type']==eye+'EyeDown')

    # Show where pupil is?
    if markpupil:
        draw = ImageDraw.Draw(im)
        landmark =next(obj for obj in fd['Landmarks'] if obj['Type']==eye+'Pupil')
        x = landmark['X'] * im.size[0];
        y = landmark['Y'] * im.size[1];
        draw.line((x , y, x , y), fill=(255, 255, 0, 128))

    cropbox=(min(max(0,el['X']*im.size[0]),im.size[0]-1),min(max(0,eu['Y']*im.size[1]),im.size[1]-1),max(0,min(im.size[0]-1,er['X']*im.size[0])),max(0,min(im.size[1],ed['Y']*im.size[1])))
    im=im.crop(cropbox)


    return im


