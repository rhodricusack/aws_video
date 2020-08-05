import boto3
from PIL import Image, ImageDraw
import json
import os

class RekognitionImage:
    ''' Tools for running Amazon Rekognition on images and annotating those images with the results
    '''

    def __init__(self,fn,vidpos=None):
        # Provide with image and time of this image
        self.fn= fn
        self.infant_age_threshold=10
        self.client = boto3.client('rekognition',region_name='eu-west-1')
        self.myfill=(0,0,1)
        self.vidpos=vidpos

    def rekognition(self):
        # Use file extension to determine whether to run afresh or load previous JSON
        filename, file_extension = os.path.splitext(self.fn)

        # Run rekognition
        with open(self.fn, 'rb') as img:
            # Get Rekognition to annotate the faces
            self.response = self.client.detect_faces(Attributes=["ALL"], Image={'Bytes': img.read()})
        return self.response

    def markfaces(self,im=None,cols=None):
        if im is None:
            im=Image.open(self.fn)

        # Default colour
        linecol=self.myfill

        # Annotate image
        draw = ImageDraw.Draw(im)
        # Mark each face
        for ind,fd in enumerate(self.response['FaceDetails']):
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

    def marklandmarks(self,im=None):
        if im is None:
            im=Image.open(self.fn)

        # Annotate image
        draw = ImageDraw.Draw(im)

        for fd in self.response['FaceDetails']:
            for landmark in fd['Landmarks']:
                x=landmark['X']*im.size[0];
                y=landmark['Y']*im.size[1];
                draw.point((x, y),fill=(255, 255, 0, 192))

        return im


    def markeyesclosed(self,im=None):
        if im is None:
            im=Image.open(self.fn)

        # Annotate image
        draw = ImageDraw.Draw(im)

        for fd in self.response['FaceDetails']:
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


    def geteye(self,eye,fd=None,im=None,markpupil=False):
        if not eye in ['left','right']:
            raise 'Specify left or right eye'

        if im is None:
            im = Image.open(self.fn)

        if fd is None:
            if len(self.response['FaceDetails'])>1:
                raise('More than one face found in image and which one not specified')
            else:
                fd=self.response['FaceDetails'][0]

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

    def save(self,fn):
        dets={'response':self.response,'fn':self.fn,'vidpos':self.vidpos}
        with open(fn, 'w') as outfile:
            json.dump(dets, outfile)

    def load(self,fn):
        with open(fn) as infile:
            inp=json.load(infile)
            self.response=inp['response']
            self.fn=inp['fn']
            self.vidpos=inp['vidpos']

    def find_infant_faces(self):
        '''Find index and age of youngest face
        Inputs: 
        Returns: index list corresponding to faces where age < self.infant_age_threshold
        '''
        infant_ind= [x[0] for x in enumerate(self.response['FaceDetails']) if x[1]['AgeRange']['Low']<self.infant_age_threshold]
        return infant_ind

