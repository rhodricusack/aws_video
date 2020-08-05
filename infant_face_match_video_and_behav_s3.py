import os
import numpy as np
import loadlookingtimes
import boto3
import experiment
from matching_s3_objects import get_matching_s3_keys


def find_behav_for_video(vid):
    '''
    Given a video filename, identifies corresponding behavioural files, and prepares the experimental settings
    Inputs: vidfn - path to s3 file
    Outputs: a dictionary with
        ['matches']=corresponding behavioural files
        ['experiment']=corresponding experiment object
    Rhodri Cusack, Trinity College Dublin, 2018-02-11, rhodri@cusacklab.org
    Lookit data from Kim Scott, MIT via osf.io
    '''

    resp={}

    bucketname=vid['S3Bucket'] # S3 bucket that contains our data
    vidfn=vid['S3ObjectName']

    ctspath='osf/mbcu2/osfstorage/Coding test studies' # Base path in S3 bucket for OSF data

    # Find out which experiment and direct to correct behavioural path
    (head, tail) = os.path.split(vidfn)
    if tail.startswith('oneshot'):
        mancodpth = '/'.join([ctspath, 'Oneshot'])
        resp['experiment']=experiment.LookingTime
    elif  tail.startswith('novelverbs'):
        mancodpth = '/'.join([ctspath, 'Novelverbs'])
        resp['experiment']=experiment.PreferenceLooking
    else:
        mancodpth= 'Roni/vcod'
        resp['experiment'] = experiment.LookingTime

    # Match the each video filename to the corresponding manual coding filename[s]
    # For each video filename find all matching manual coding filenames, and create key we can use for this subject
    flds = tail.split('_')
    if len(flds)>=3:
        subjname = '%s_%s%s' % (flds[0], flds[1], flds[2])
    else:
        flds=flds[0].split('.')
        subjname = flds[0]
    matches = []
    prefix="/".join([mancodpth,subjname])
    resp['matches']=[f for f in get_matching_s3_keys(bucketname,prefix=mancodpth,suffix='.txt') if os.path.basename(f).startswith(subjname)]
    resp['key'] = ''.join([x if x.isalnum() else '_' for x in subjname])
    resp['S3Bucket']=bucketname

    return resp


# Testing
if __name__ == "__main__":
    testlist=[['Roni','s3://infantrekognition/31IBVUNM9SYPJVQ0RMZ56CS4V5ZVFZ.mp4'],
           ['novelverbs', 's3://infantrekognition/osf/dqmcv/osfstorage/novelverbs_coded/novelverbs_1128_child0_DEH_free.mp4'],
           ['oneshot',    'https://s3-eu-west-1.amazonaws.com/infantrekognition/osf/dqmcv/osfstorage/oneshot_coded/oneshot_1171_child0_44%7DhgV_free.mp4']]
    for test in testlist:
        print("Test of %s with file %s"%(test[0],test[1]))
        print("Match is %s"%find_behav_for_video(test[1]))