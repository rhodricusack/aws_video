from infant_face_match_video_and_behav_s3 import find_behav_for_video
import videotools
import os
import boto3
import numpy as np
from pathlib import Path
from scipy import stats
import tempfile
import skvideo
import faceannotation
import experiment
import s3tools
import json
import pickle
from sklearn import metrics

def process_rekognition_video(compmsg):
    # For behavioural data
    bucket='infant_rekognition'

    # Get result from rekognition
    rekognition = boto3.client('rekognition')
    jobid=compmsg['JobId']
    response = rekognition.get_face_detection(JobId=jobid)
    assert response['JobStatus']=='SUCCEEDED', "Rekogntion job status not SUCCEEDED but %s"%response['JobStatus']
    allfaces = response['Faces']
    while 'NextToken' in response:
        response = rekognition.get_face_detection(JobId=jobid, NextToken=response['NextToken'])
        allfaces.extend(response['Faces'])
    print("%d faces detected"%len(allfaces))

    # Work out what sampling rekogition seems to be using
    ts= [face['Timestamp'] for face in allfaces]
    difft=np.ediff1d(ts)
    difft=[x for x in difft if not x==0]
    deltat=stats.mode(difft).mode
    print("Delta t is %f"%deltat)

    # Get the behavioural file that corresponds to the video
    vid=compmsg['Video']
    behav=find_behav_for_video(vid)
    exp=[]
    for rater in behav['matches']:
        pth=s3tools.getpath({'S3Bucket':behav['S3Bucket'],'S3ObjectName':rater})
        exp.append(behav['experiment'](pth))

    print(behav)

    # Get the video
    v = videotools.Video(vid)
    if v._pth is None or not os.path.exists(v._pth):
        print("Video not found")
        return False
    else:
        v.open()
        dur = v.get_dur()
        fps = v.get_fps()
        print("Dur %f and FPS %f"%(dur,fps))

        timestamps=[face['Timestamp'] for face in allfaces]

        # Annotated video filename
        outkey_annotated="annotated/" + vid['S3ObjectName']
        outfn_annotated=s3tools.getcacheoutpath(outkey_annotated)

        # Coding filename
        outkey_coding = "coding/" + os.path.splitext(vid['S3ObjectName'])[0] + '.pickle'
        outfn_coding = os.path.join(Path.home(), ".s3cache-out", outkey_coding)
        # Make directory if necessary
        dirname = os.path.dirname(outfn_coding)
        if not os.path.exists(dirname):
            os.makedirs(dirname)


        writer = skvideo.io.FFmpegWriter(outfn_annotated)

        facetimes=[face['Timestamp'] for face in allfaces]

        firstface=0
        lastface=0

        # Store all coding results
        coding=[]

        while v.isopen or v.currtime>10:
            # Get frame
            img=v.get_next_frame()

            # End of video
            if img is None:
                break

            currtime_ms=np.round(v.currtime*1000)

            # Move forwards the first face we need to consider, if appropriate
            while firstface<len(facetimes) and facetimes[firstface]<(currtime_ms-deltat/2 + 1): # 1 ms buffer for rounding errors
                firstface+=1


            # Add all faces to the last one we need to consider
            faces=[]
            for ind in range(firstface,len(facetimes)-1):
                if facetimes[ind]>(currtime_ms+deltat/2):
                    break
                faces.append(ind)

            # Count them and set up colours
            countfaces = len(faces)
            cols = [(0, 0, 255, 128)] * countfaces

            # Mark one or more infant faces in green
            infantfaces=[]
            infantind=[]
            for i0, faceind in enumerate(faces):
                if allfaces[faceind]['Face']['AgeRange']['Low']<10:
                    infantind.append(i0) # which of elements in faces are infants
                    infantfaces.append(allfaces[faceind]['Face'])
                    cols[i0] = (255, 0, 0, 128)

            # Automatic scoring
            autocod=exp[0].score_face(infantfaces)
            if len(infantind)==1:
                cols[infantind[0]]=autocod['colour']

            # Annotate faces
            img = faceannotation.markfaces([allfaces[item] for item in faces],img, cols)
            img = faceannotation.marklandmarks([allfaces[item] for item in faces],img)
            img = faceannotation.markeyesclosed([allfaces[item] for item in faces],img)

            # Get manual coding average
            # Annotate manual coding status on border of image
            mancod_allraters = []
            coltot = (0, 0, 0, 0)
            for singlerater in exp:
                mancod_allraters.append(singlerater.get_mancod_state(currtime_ms))
                coltot = np.array(coltot) + np.array(mancod_allraters[-1]['colour'])
            colmean = tuple(map(int,coltot / len(exp)))

            # Outer border, coloured appropriately
            img = faceannotation.markmanual(img,colmean)

            # Write annotated frame
            writer.writeFrame(img)

            # Store the coding results
            coding.append({'autocod': autocod, 'mancod_allraters': mancod_allraters,'faces':faces,'infantind':infantind})


        writer.close()

        # Frame-by-frame manual coding - modal value across raters
        m=[stats.mode([oneperson['code'] for oneperson in c['mancod_allraters'] if not oneperson['code'] is None ]) for c in coding]
        m=[x.mode[0] if len(x.mode)>=1 else None for x in m] # Correct for frames with no ratings
        m=[0 if x is None else x for x in m] # No ratings, set code to zero

        # Frame-by-frame auto coding
        a=[c['autocod']['code'] for c in coding]
        a=[0 if x is None else x for x in a] # No faces, set code to zero

        # Possible codes and descriptions
        possible_codes=exp[0].possible_codes()

        # Calculate confusion matrix
        conf = metrics.confusion_matrix(m,a,labels=possible_codes['code'])
        print(exp[0].possible_codes()['desc'])
        print(conf)

        # Create summary dict
        summary={'coding':coding,'conf':conf,'possible_codes':possible_codes,'allfaces':allfaces,'behav':behav,'vid':vid,'compmsg':compmsg}

        # Write coding file
        with open(outfn_coding, 'wb') as f:
            pickle.dump(summary, f)

        # Write annotated video and coding files to s3
        s3 = boto3.resource('s3')
        s3.Bucket(vid['S3Bucket']).upload_file(outfn_annotated,outkey_annotated)
        s3.Bucket(vid['S3Bucket']).upload_file(outfn_coding,outkey_coding)

    return True

# Testing
if __name__ == "__main__":
    # Get all of the waiting jobs
    sqs=boto3.resource('sqs')
    q=sqs.get_queue_by_name(QueueName='anothersqs')
    all_messages=[]
    rs = q.receive_messages()
    while len(rs) > 0:
        all_messages.extend(rs)
        rs = q.receive_messages()
    print("Got %d messages"%len(all_messages))

    # Process them all
    for message in all_messages:
        print(message.body)
        j=json.loads(message.body)
        print(j)
        jm=json.loads(j['Message'])
        process_rekognition_video(jm)
