import os
import glob
import numpy as np
import math
import json
import rekognitiontools


class Experiment:
    """
    Base class for a single subject in an experiment
    Classes derived from this one handle the logic specific to an experiment, like reading in manual coding file
    using it to create the state at a momemnt in time, and then scoring a detected infant face for a moment in time

    Attributes:
        filename - filename with path for manual coding
        mancod - manual coding
        mancod_header - header for manual coding
    """

    def __init__(self, filename):
        self.mancod = {}
        self.mancod_header = {}
        self.filename = filename
        self.read_mancod()

    def read_mancod(self):
        # Function to read in the manual coding file
        return None

    def get_mancod_state(self, vidpos):
        # Get manual coding state at a corresponding time in the movie vidpos (measured in seconds)
        return None

    def score_face(self, vidpos_ms, facedetail):
        # Score a face - returns True (automatic coding correct) or False (automatic coding incorrect)
        return None

    def possible_codes(self):
        return None

    def read_mancod_vcode(self):
        # Reads in a csv file in vcode format, as used by lookit
        with open(self.filename, 'r') as fobj:
            isheader = True
            items = []
            for lne in fobj:
                if not lne.strip():
                    isheader = False
                else:
                    if isheader:
                        fldnme = 'columns'
                        fld = ''
                        for char in lne:
                            if char == ':':
                                # Allocate items to previous field
                                if items:
                                    self.mancod_header[fldnme] = items
                                    items = []

                                # Get new fieldname
                                fldnme = fld
                                fld = ''
                            elif char == ',' or char == '\n':
                                # Found a comma delimited item
                                items.append(fld)
                                fld = ''
                            else:
                                # Just a character in a field name or field
                                fld = fld + char
                        # Add last item
                        if fld:
                            items.append(fld)
                        # Allocate last items to header
                        if items:
                            self.mancod_header[fldnme] = items
                            items = []
                    else:
                        flds = lne.strip().split(',')
                        for ind, col in enumerate(self.mancod_header['columns']):
                            if not col in self.mancod:
                                self.mancod[col] = []
                            self.mancod[col].append(flds[ind])
        return True


class LookingTime(Experiment):
    '''
    Experiment in which infant looks or doesn't look at camera
    Manual coding is with extended periods coded in CSV with onset and duration
    '''

    def __init__(self, filename):
        Experiment.__init__(self, filename)

    def read_mancod(self):
        # This experiment uses vcode csv format
        return self.read_mancod_vcode()

    def get_mancod_state(self, vidpos_ms):
        ons = [float(x) for x in self.mancod['Time']]
        dur = [float(x) for x in self.mancod['Duration']]
        evtype = self.mancod['TrackName']

        # Was the infant marked as in a looking phase?
        ind = [x[0] for x in enumerate(ons) if
               evtype[x[0]].lower() == 'looking' and x[1] <= vidpos_ms and vidpos_ms <=(x[1] + dur[x[0]])]
        if ind:
            behav = {'code': 2, 'desc': 'looking', 'colour': (0, 255, 0, 255)}
        else:
            behav = {'code': 1, 'desc': 'not looking', 'colour': (255, 0, 0, 255)}

        return behav

    def score_face(self, facedetaillist):
        # Auto coding
        # Inputs: facedetail
        # Returns: [numeric code, string code]
        if not len(facedetaillist) == 1:
            autocod = {'code': 0, 'desc': 'notoneface', 'colour': (255, 0, 255, 255)}
        else:
            facedetail = facedetaillist[0]
            pose = facedetail['Pose']
            rot=self.eulerToRotationAxisAngle(pose['Yaw'],pose['Pitch'],pose['Roll'],units='degrees')
            angoff = rot[1]
            autocod = {'code': 1, 'desc': 'notlooking', 'colour': (255, 0, 0, 255)}
            if angoff < 20:
                autocod = {'code': 2, 'desc': 'looking', 'colour': (0, 255, 0, 255)}
        return autocod


    def possible_codes(self):
        return {'code':[0,1,2],'desc': ['none','notlooking','looking']}

    def eulerToRotationAxisAngle(self,yaw,pitch,roll,units='radians'):
        if units=='degrees':
            yaw=math.radians(yaw)
            pitch=math.radians(pitch)
            roll=math.radians(roll)

        # Thanks to Zacobria Lars Skovgaard https://plus.google.com/u/0/116832821661215606670?rel=author
        # for this post http://www.zacobria.com/universal-robots-knowledge-base-tech-support-forum-hints-tips/python-code-example-of-converting-rpyeuler-angles-to-rotation-vectorangle-axis-for-universal-robots/
        yawMatrix = np.matrix([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        pitchMatrix = np.matrix([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        rollMatrix = np.matrix([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        R = yawMatrix * pitchMatrix * rollMatrix

        theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
        multi = 1 / (2 * math.sin(theta))

        rx = multi * (R[2, 1] - R[1, 2]) * theta
        ry = multi * (R[0, 2] - R[2, 0]) * theta
        rz = multi * (R[1, 0] - R[0, 1]) * theta

        if units=='degrees':
            theta=math.degrees(theta)

        return [[rx,ry,rz],theta]


class RoniLookingTime(LookingTime):
    '''Roni looking time data doesn't come from a file but is passed as a list of durations and onsets
    Convert times to ms from s, to match vcode coding scheme
    '''
    def read_mancod(self):
        self.mancod['Time']=[x[0]*1000 for x in self.filename]
        self.mancod['Duration']=[(x[1]-x[0])*1000 for x in self.filename]
        self.mancod['TrackName']=['looking' for x in self.filename]

class PreferenceLooking(Experiment):
    '''
    Experiment in which infant looks left or right.
    Manual coding is like novelwords from lookit.com, so coding events (with zero duration) for left, right,
    and extended events for looking (centrally)
    '''

    def __init__(self, filename):
        Experiment.__init__(self, filename)

    def read_mancod(self):
        # This experiment uses vcode csv format
        return self.read_mancod_vcode()

    def get_mancod_state(self, vidpos_ms):
        '''
        Behavioural decoding for lookit's novelverbs
        Three possible behavioural codes: looking (forwards), left or right
        Inputs: vidpos_ms is in ms
        Returns: [numeric code, string code]
            numeric code= None | -1 | 1 | 0
            string code = None | left | right | away | looking
            col code= black | green | red | magenta | yellow
        '''

        ons = [float(x) for x in self.mancod['Time']]
        dur = [float(x) for x in self.mancod['Duration']]
        evtype = self.mancod['TrackName']

        # Returned if not looking and no left/right previous state
        behav = {'code': 0, 'desc': 'none', 'colour': (0, 0, 0, 255)}

        # Was the infant marked as in a looking phase?
        ind = [x[0] for x in enumerate(ons) if
               evtype[x[0]] == 'looking' and x[1] <= vidpos_ms and vidpos_ms <=(x[1] + dur[x[0]]) ]
        if ind:
            behav = {'code': 0, 'desc': 'looking', 'colour': (255, 255, 0, 255)}
        else:
            # Not looking, so was last judgement left, right or away?
            leftrightind = [x[0] for x in enumerate(evtype) if
                            x[1] == 'left' or x[1] == 'right' or x[1] == 'away']  # indices of left or right events
            pastind = [x for x in leftrightind if
                       ons[x] < vidpos_ms]  # indices of subset of these events earlier than vidpos

            if pastind:
                lastevent = np.argmax([ons[x] for x in pastind])  # most recent of these
                if evtype[pastind[lastevent]] == 'left':
                    behav = {'code': -1, 'desc': 'left', 'colour': (0, 255, 0, 255)}
                elif evtype[pastind[lastevent]] == 'right':
                    behav = {'code': 1, 'desc': 'right', 'colour': (255, 0, 0, 255)}
                elif evtype[pastind[lastevent]] == 'away':
                    behav = {'code': 2, 'desc': 'away', 'colour': (255, 0, 255, 255)}
                elif evtype[pastind[lastevent]] == 'outofframe':
                    behav = {'code': 3, 'desc': 'outofframe', 'colour': (0, 0, 0, 255)}

        return behav

    def possible_codes(self):
        return {'code':[-1,0,1,2,3],'desc': ['left','none','right','away','outofframe']}


    def score_face(self, facedetaillist):
        '''
        Scores a face into left vs. right
        Inputs: facedetaillist for infant faces
        Returns: [numeric code, string code, colour code]
            numeric code= None | -1 | 1 | 0
            string code = None | left | right | looking
            colour code = suggested colour for rendering this option
        '''

        if not len(facedetaillist) == 1:
            autocod = {'code': None, 'desc': 'notoneface', 'colour': (255, 0, 255, 255)}

        else:
            facedetail = facedetaillist[0]
            # Do automatic coding
            pose = facedetail['Pose']
            #            yaw = pose['Yaw']
            eyesopen = facedetail['EyesOpen']['Value'] or facedetail['EyesOpen']['Confidence'] < 90

            left_pupil_offset = [x for x in facedetail['Landmarks'] if x['Type'] == 'leftPupil'][0]['X'] - \
                                [x for x in facedetail['Landmarks'] if x['Type'] == 'eyeLeft'][0]['X']
            right_pupil_offset = [x for x in facedetail['Landmarks'] if x['Type'] == 'rightPupil'][0]['X'] - \
                                 [x for x in facedetail['Landmarks'] if x['Type'] == 'eyeRight'][0]['X']
            av_pupil_offset = (left_pupil_offset + right_pupil_offset) / 2

            # Never returns "looking" with this logic
            if not eyesopen:
                autocod = {'code': None, 'desc': None, 'colour': (255, 0, 255, 255)}
            else:
                if av_pupil_offset <= 0:
                    autocod = {'code': -1, 'desc': 'left', 'colour': (0, 255, 0, 255)}
                else:
                    autocod = {'code': 1, 'desc': 'right', 'colour': (255, 0, 0, 255)}

        return autocod


# Examples
if __name__ == "__main__":
    exp = PreferenceLooking('c:/Users/Rhodri Cusack/Google Drive/Lookit/Novelverbs/Joseph/novelverbs_-2child0.txt')
    # Manual coding
    print(exp.get_mancod_state(20000))

    # Auto coding
    fd = rekognitiontools.RekognitionImage(
        'C:/Users/Rhodri Cusack/Documents/infant_face_recognition/oneshot_568_child0/measures_singlefr002000.json')
    fd.rekognition()
    print(exp.score_face(fd.find_infant_faces()))
