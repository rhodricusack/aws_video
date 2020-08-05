import hashlib
from infant_face_match_video_and_behav_s3 import find_behav_for_video
import videotools
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from scipy import stats
import skvideo
import faceannotation
import s3tools
import json
import pickle
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import matplotlib.pyplot as plt


def run_video_rekognition(bucket, prefix, doevenifdone=False):
    """
    Runs AWS face rekognition on all of the files in a directory
    """

    # List all objects in the bucket
    s3 = boto3.client('s3')
    allfiles = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # For testing just one file
    # allfiles['Contents']=[allfiles['Contents'][2]]

    # Create mediaconvert object
    rekognition = boto3.client('rekognition')

    # For each file create start face detection job
    for myobj in allfiles['Contents']:
        # Only process objects with .mp4 suffix
        if myobj['Key'][-4:] == '.mp4':
            print(myobj['Key'])
            filename = myobj['Key']

            # Don't submit if already coded
            outkey_coding = "coding/" + os.path.splitext(myobj['Key'])[0] + '.pickle'

            outputpresent = s3.list_objects(Bucket=bucket, Prefix=outkey_coding)

            if doevenifdone or 'Contents' not in outputpresent:
                response = rekognition.start_face_detection(
                    Video={'S3Object': {
                        'Bucket': bucket, 'Name': filename}
                    },
                    ClientRequestToken=hashlib.md5(str('s3://' + bucket + '/' + filename).encode()).hexdigest(),
                    FaceAttributes='ALL',
                    NotificationChannel={
                        'SNSTopicArn': 'arn:aws:sns:eu-west-1:807820536621:AmazonRekognition_cusack_infant',
                        'RoleArn': 'arn:aws:iam::807820536621:role/cusack_infant_rekognition'
                    }
                )
                print(response['JobId'])
                print(response)
            else:
                print("Already done %s" % outkey_coding)


def process_sqs_responses(bucket, sqsqueuename, doevenifdone=False):
    """
    When rekognition finishes running, it posts to an SNS which I have configured to write to an SQS queue
    This script reads the queue, loads the results of rekognition, and formats them into a python structure
    :param bucket: bucket for output data
    :param sqsqueuename: name of SQS queue containing results
    :param doevenifdone: do it again even if output already present
    :return:
    """

    sqs = boto3.resource('sqs')
    q = sqs.get_queue_by_name(QueueName=sqsqueuename)
    all_messages = []
    rs = q.receive_messages()
    while len(rs) > 0:
        all_messages.extend(rs)
        rs = q.receive_messages()
    print("Got %d messages" % len(all_messages))

    # Process them all
    for message in all_messages:
        print(message.body)
        j = json.loads(message.body)
        print(j)
        jm = json.loads(j['Message'])
        process_rekognition_video(bucket, jm, doevenifdone=doevenifdone)


def reprocess_video(bucket, prefix):
    """
    Script for re-running the initial reprocessing stages, which doesn't need SQS messages, but instead scans videos
    """
    # List all objects in the bucket
    s3 = boto3.client('s3')
    allfiles = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # For each file create start face detection job
    for myobj in allfiles['Contents']:
        # Only process objects with .mp4 suffix
        if myobj['Key'][-4:] == '.mp4':
            vid = {'S3ObjectName': myobj['Key'], 'S3Bucket': bucket}
            compmsg = {'Video': vid, 'JobId': None}

            # Coding filename
            key_coding = "coding/" + os.path.splitext(vid['S3ObjectName'])[0] + '.pickle'
            fn_coding = s3tools.getcacheinpath(key_coding)
            if os.path.exists(fn_coding):
                with open(fn_coding, 'rb') as f:
                    obj = pickle.load(f)

                compmsg['allfaces'] = obj['allfaces']

                process_rekognition_video(bucket, compmsg, doevenifdone=True)


def process_rekognition_video(bucket, compmsg, doevenifdone=False):
    """
    Extract details from processed video,
    :param bucket: S3 bucket for data
    :param compmsg: SQS message returned by rekogntion
    :param doevenifdone: Do again even if output already present
    :return:
    """

    # Get result from rekognition
    rekognition = boto3.client('rekognition')
    jobid = compmsg['JobId']
    vid = compmsg['Video']

    if vid['S3ObjectName'][-12:] == '_lighter.mp4':
        print("Skipping lighter video %s" % vid['S3ObjectName'])
        return

    # Annotated video filename
    outkey_annotated = "annotated/" + vid['S3ObjectName']
    outfn_annotated = s3tools.getcacheoutpath(outkey_annotated)

    # Coding filename
    outkey_coding = "coding/" + os.path.splitext(vid['S3ObjectName'])[0] + '.pickle'
    outfn_coding = os.path.join(Path.home(), ".s3cache-out", outkey_coding)

    s3bucket = boto3.resource('s3').Bucket(vid['S3Bucket'])
    s3client = boto3.client('s3')
    if doevenifdone or  'Contents' not in s3client.list_objects(Bucket=vid['S3Bucket'], Prefix=outkey_coding):
        try:
            if jobid is not None:
                response = rekognition.get_face_detection(JobId=jobid)

                assert response['JobStatus'] == 'SUCCEEDED', "Rekogntion job status not SUCCEEDED but %s" % response[
                    'JobStatus']

                allfaces = response['Faces']
                while 'NextToken' in response:
                    response = rekognition.get_face_detection(JobId=jobid, NextToken=response['NextToken'])
                    allfaces.extend(response['Faces'])
                print("%d faces detected" % len(allfaces))
            else:
                allfaces = compmsg['allfaces']

            # Work out what sampling rekogition seems to be using
            ts = [face['Timestamp'] for face in allfaces]
            difft = np.ediff1d(ts)
            difft = [x for x in difft if not x == 0]
            deltat = stats.mode(difft).mode
            print("Delta t is %f" % deltat)

            # Get the behavioural file that corresponds to the video
            behav = find_behav_for_video(vid)

            if not behav['matches']:
                print("No behavioural file found to correspond to %s" % vid['S3ObjectName'])
                return False

            exp = []
            for rater in behav['matches']:
                pth = s3tools.getpath({'S3Bucket': behav['S3Bucket'], 'S3ObjectName': rater})
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
                print("Dur %f and FPS %f" % (dur, fps))

                timestamps = [face['Timestamp'] for face in allfaces]

                # Make directory if necessary
                dirname = os.path.dirname(outfn_coding)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                writer = skvideo.io.FFmpegWriter(outfn_annotated)

                facetimes = [face['Timestamp'] for face in allfaces]

                firstface = 0
                lastface = 0

                # Store all coding results
                coding = []

                while v.isopen:
                    # Get frame
                    img = v.get_next_frame()

                    # End of video
                    if img is None:
                        break

                    currtime_ms = np.round(v.currtime * 1000)

                    # Move forwards the first face we need to consider, if appropriate
                    while firstface < len(facetimes) and facetimes[firstface] < (
                            currtime_ms - deltat / 2 + 1):  # 1 ms buffer for rounding errors
                        firstface += 1

                    # Add all faces to the last one we need to consider
                    faces = []
                    for ind in range(firstface, len(facetimes) - 1):
                        if facetimes[ind] >= (currtime_ms + deltat / 2):
                            break
                        faces.append(ind)

                    # Count them and set up colours
                    countfaces = len(faces)
                    cols = [(0, 0, 255, 128)] * countfaces

                    # Mark one or more infant faces in green
                    infantfaces = []
                    infantind = []
                    for i0, faceind in enumerate(faces):
                        if allfaces[faceind]['Face']['AgeRange']['Low'] < 10:
                            infantind.append(i0)  # which of elements in faces are infants
                            infantfaces.append(allfaces[faceind]['Face'])
                            cols[i0] = (255, 0, 0, 128)

                    # If two largely overlapping faces are found, they must be the same one, so delete one
                    if len(infantind) == 2:
                        bb0 = allfaces[faces[infantind[0]]]['Face']['BoundingBox']
                        bb1 = allfaces[faces[infantind[1]]]['Face']['BoundingBox']
                        dx = bb0['Left'] - bb1['Left']
                        mw = 0.5 * (bb0['Width'] + bb1['Width'])
                        dy = bb0['Top'] - bb1['Top']
                        mh = 0.5 * (bb0['Height'] + bb1['Height'])

                        if np.sqrt((dx / mw) ** 2 + (dy / mh) ** 2) < 0.1:  # shifted by less than 10% of size
                            infantind = [infantind[0]]
                            infantfaces = [infantfaces[0]]

                    # Automatic scoring
                    autocod = exp[0].score_face(infantfaces)
                    if len(infantind) == 1:
                        cols[infantind[0]] = autocod['colour']

                    # Annotate faces
                    img = faceannotation.markfaces([allfaces[item] for item in faces], img, cols)
                    img = faceannotation.marklandmarks([allfaces[item] for item in faces], img)
                    img = faceannotation.markeyesclosed([allfaces[item] for item in faces], img)

                    # Get manual coding average
                    # Annotate manual coding status on border of image
                    mancod_allraters = []
                    coltot = (0, 0, 0, 0)
                    for singlerater in exp:
                        mancod_allraters.append(singlerater.get_mancod_state(currtime_ms))
                        coltot = np.array(coltot) + np.array(mancod_allraters[-1]['colour'])
                    colmean = tuple(map(int, coltot / len(exp)))

                    # Outer border, coloured appropriately
                    img = faceannotation.markmanual(img, colmean)

                    # Write annotated frame
                    writer.writeFrame(img)

                    # Store the coding results
                    coding.append({'autocod': autocod, 'mancod_allraters': mancod_allraters, 'faces': faces,
                                   'infantind': infantind})

                writer.close()

                # Frame-by-frame manual coding - modal value across raters
                m = [stats.mode(
                    [oneperson['code'] for oneperson in c['mancod_allraters'] if not oneperson['code'] is None]) for c
                     in coding]
                m = [x.mode[0] if len(x.mode) >= 1 else None for x in m]  # Correct for frames with no ratings
                m = [0 if x is None else x for x in m]  # No ratings, set code to zero

                # Frame-by-frame auto coding
                a = [c['autocod']['code'] for c in coding]
                a = [0 if x is None else x for x in a]  # No faces, set code to zero

                # Possible codes and descriptions
                possible_codes = exp[0].possible_codes()

                # Calculate confusion matrix
                conf = metrics.confusion_matrix(m, a, labels=possible_codes['code'])
                print(exp[0].possible_codes()['desc'])
                print(conf)

                # Create summary dict
                summary = {'coding': coding, 'conf': conf, 'possible_codes': possible_codes, 'allfaces': allfaces,
                           'behav': behav, 'vid': vid, 'compmsg': compmsg, 'deltat': deltat, 'fps': fps, 'dur': dur}

                # Write coding file
                with open(outfn_coding, 'wb') as f:
                    pickle.dump(summary, f)

                # Write annotated video and coding files to s3
                s3bucket.upload_file(outfn_annotated, outkey_annotated)
                s3bucket.upload_file(outfn_coding, outkey_coding)

                return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print("No response from rekognition available for jobid %s" % compmsg['JobId'])
                return False
            else:
                raise
    else:
        print("Not repeating previously annotated %s" % outkey_coding)
        return False


def post_rekognition_summary(bucket, experimentfilter):
    """
    Aggregates data across subjects from auto and manual coding
    Rhodri Cusack Trinity College Dublin 2018-03-11 rhodri@cusacklab.org
    :param bucket: s3 bucket for auto coding to load
    :param experimentfilter: experiment to work on (path in s3 for coded results)
    :return:
    """
    # Write annotated video and coding files to s3
    s3 = boto3.resource('s3')
    s3bucket = s3.Bucket(bucket)

    # Columns to be extracted
    pose_columns = ['Pitch', 'Yaw', 'Roll']
    landmark_columns = [['leftPupil', 'X'], ['rightPupil', 'X'], ['eyeLeft', 'X'], ['eyeRight', 'X']]
    bb_columns = ['Top', 'Left', 'Width', 'Height']

    df = []

    allagerangelow = []
    allagerangehigh = []

    # Download each subject and add to dataframe
    for codobj in s3bucket.objects.filter(Prefix=experimentfilter):
        # This is where the output will be written
        outkey = os.path.join('coding_summary', experimentfilter, 'summary.pickle')

        fn = s3tools.getpath({'S3Bucket': bucket, 'S3ObjectName': codobj.key})
        df.append({'S3Bucket': bucket, 'S3ObjectName': codobj.key,
                   'df': pd.DataFrame(columns=pose_columns + [x[0] for x in landmark_columns] + ['mancod'])})
        with open(fn, 'rb') as f:
            obj = pickle.load(f)

            df[-1]['deltat'] = obj['deltat']
            df[-1]['fps'] = obj['fps']
            df[-1]['dur'] = obj['dur']

            q = [[x['faces'][y] for y in x['infantind']] for x in obj[
                'coding']]  # "faces" contains indices within allfaces. "infantind" contains indices within faces. Get indices of infants among allfaces.
            myfaces = [[obj['allfaces'][z1] for z1 in z0] for z0 in q]  # get actual face data from allfaces

            # Get the face details
            allagerangelow.extend([x['Face']['AgeRange']['Low'] for x in obj['allfaces']])
            allagerangehigh.extend([x['Face']['AgeRange']['High'] for x in obj['allfaces']])

            for ind, item in enumerate(myfaces):
                # Build a row
                row = {}

                # One infant face?
                if len(item) == 1:
                    # Pose
                    for col in pose_columns:
                        row[col] = item[0]['Face']['Pose'][col]
                    # Landmarks
                    for col in landmark_columns:
                        row[col[0]] = [x[col[1]] for x in item[0]['Face']['Landmarks'] if x['Type'] == col[0]][0]
                    # Eyes open
                    row['EyesOpenValue'] = int(item[0]['Face']['EyesOpen']['Value'])
                    row['EyesOpenConfidence'] = item[0]['Face']['EyesOpen']['Confidence']
                    # Bounding box
                    for col in bb_columns:
                        row['BoundingBox' + col] = item[0]['Face']['BoundingBox'][col]
                    # Confidence and Quality
                    row['Confidence'] = item[0]['Face']['Confidence']
                    row['QualitySharpness'] = item[0]['Face']['Quality']['Sharpness']
                    row['QualityBrightness'] = item[0]['Face']['Quality']['Brightness']

                # And mean manual coder
                list = [x['code'] for x in obj['coding'][ind]['mancod_allraters']]
                row['mancod'] = float(max(set(list), key=list.count))
                df[-1]['df'] = df[-1]['df'].append(pd.DataFrame([row]), ignore_index=True)

        print(df[-1]['df'].describe())

    # Save result and upload to S3
    fname = s3tools.getcacheoutpath(outkey)
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).upload_file(fname, outkey)

    # Done!
    print("All done")

    return ({'agerangelow': allagerangelow, 'agerangehigh': allagerangehigh})


def run_machine_learning(bucket, experimentfilters, outpth, colorscheme=None, possible_codes=None, usemedianforcentering=False):
    """
    Runs leave-one-subject-out machine learning

    :param bucket: bucket to work
    :param experimentfilters: list of paths to coding_summary
    :return:
    """
    # Colorscheme
    if not colorscheme:
        colorscheme = ['r', 'g', 'b']

    # Write annotated video and coding files to s3
    s3 = boto3.resource('s3')
    s3bucket = s3.Bucket(bucket)

    classifycols = ['Pitch', 'Roll', 'Yaw', 'leftPupil', 'rightPupil', 'eyeLeft', 'eyeRight', 'EyesOpenValue',
                    'EyesOpenConfidence']

    # All experiments
    allmldf = pd.DataFrame()

    # Load summary file
    allpred = []
    for spind, sp in enumerate(experimentfilters):
        fn = s3tools.getpath({'S3Bucket': bucket, 'S3ObjectName': os.path.join(sp, 'summary.pickle')})
        with open(fn, 'rb') as f:
            obj = pickle.load(f)

            # Find blanks
            for testsubjind, testdata in enumerate(obj):
                obj[testsubjind]['df_dropna'] = testdata['df'].dropna()

            # Centre the classification columns
            for testsubjind, testdata in enumerate(obj):
                if usemedianforcentering:
                    testdata['df_zerocentre'] = testdata['df_dropna'][classifycols].subtract(
                        testdata['df_dropna'][classifycols].median())
                else:
                    testdata['df_zerocentre'] = testdata['df_dropna'][classifycols].subtract(
                        testdata['df_dropna'][classifycols].mean())

            # Leave-one-subject-out classification
            pred = []
            mldf = pd.DataFrame()
            for testsubjind, testsubj in enumerate(obj):
                testlabels = testsubj['df_dropna']['mancod']
                testfeat = testsubj['df_zerocentre']

                # Get TRAIN data from all but one subject
                trainlabels = pd.concat(
                    [x[1]['df_dropna']['mancod'] for x in enumerate(obj) if not x[0] == testsubjind])
                trainfeat = pd.concat([x[1]['df_zerocentre'] for x in enumerate(obj) if not x[0] == testsubjind])

                # Run and test classifier. Use quadratic discriminant as look/don't look is non-linear function of position
                clf = QuadraticDiscriminantAnalysis()
                clf.fit(trainfeat, trainlabels)
                prednonan = clf.predict(testfeat)

                # We've filtered out nans before predicting.
                # Put them back in before storing, so indices of pred correspond to data before filtering out nans
                predallrows = np.ones((testsubj['df'].shape[0]))
                predallrows[testsubj['df'].notnull().all(axis=1)] = prednonan
                pred.append(predallrows)

                # First index (rows) are the "truth" of manual coding in testlabels
                cnf = metrics.confusion_matrix(testlabels, prednonan, labels=possible_codes)

                # Adjust for trials in which no face was detected
                # In absence of face, machine coding defaults to option 0 - for experiment 1, no face; for experiment 2, left
                rowisnan = testsubj['df'].isnull().any(axis=1)
                cnf[0, 0] = cnf[0, 0] + (testsubj['df']['mancod'][rowisnan] == possible_codes[0]).sum()
                cnf[1, 0] = cnf[1, 0] + (testsubj['df']['mancod'][rowisnan] == possible_codes[1]).sum()

                # Signal detection theory heuristic - if hits or fa=0 then replace with half a trial, and same at max end
                fa = cnf[0, 1] if not cnf[0, 1] == 0 else 0.5
                hits = cnf[1, 1] if not cnf[1, 1] == 0 else 0.5
                n0 = cnf[0, :].sum()
                n1 = cnf[1, :].sum()
                hits = hits if not hits == n1 else n1 - 0.5
                fa = fa if not fa == n0 else n0 - 0.5
                hits = hits / n1
                fa = fa / n0

                # Summary statistics
                proponeface = len(testsubj['df_dropna']) / len(testsubj['df'])
                mldf = mldf.append(pd.DataFrame({'proponeface': proponeface,
                                                 'score': clf.score(testfeat, testlabels),
                                                 'fa': fa,
                                                 'hits': hits,
                                                 'n0': n0,
                                                 'n1': n1,
                                                 'Confidence': testsubj['df']['Confidence'].mean(skipna=True),
                                                 'QualitySharpness': testsubj['df']['QualitySharpness'].mean(
                                                     skipna=True),
                                                 'QualityBrightness': testsubj['df']['QualityBrightness'].mean(
                                                     skipna=True),
                                                 'deltat': testsubj['deltat'],
                                                 'fps': testsubj['fps'],
                                                 'dur': testsubj['dur'],
                                                 },
                                                index=[testsubj['S3ObjectName']]))
                plt.figure("boundingbox")
                ax = plt.subplot(111)
                bb = testsubj['df'][
                    ['BoundingBoxLeft', 'BoundingBoxTop', 'BoundingBoxWidth', 'BoundingBoxHeight']].mean()
                ax.add_patch(mpatches.Rectangle((bb[0], bb[1] - bb[3]), bb[2], bb[3],
                                                edgecolor=(1 - proponeface, proponeface, 0), Fill=False))
                # print(mldf.describe())

            plt.figure("ROC")
            # Add d-prime lines
            for dprime in np.arange(3):
                fan = np.arange(-5, 5 - dprime, 0.1)
                hitsn = fan + dprime
                plt.plot(norm.cdf(fan), norm.cdf(hitsn), linestyle='dashed', color='gray', alpha=0.5)
            # Add hits and fa
            plt.scatter(data=mldf, x='fa', y='hits', s=64 * mldf['proponeface'], color=colorscheme[spind])

            mldf['dprime'] = norm.ppf(mldf['hits']) - norm.ppf(mldf['fa'])
            plt.figure('Confidence-dprime')
            plt.scatter(x='Confidence', y='dprime', data=mldf, color=colorscheme[spind])
            plt.figure('QualitySharpness-dprime')
            plt.scatter(x='QualitySharpness', y='dprime', data=mldf, color=colorscheme[spind])
            plt.figure('QualityBrightness-dprime')
            plt.scatter(x='QualityBrightness', y='dprime', data=mldf, color=colorscheme[spind])
            plt.figure('proponeface-dprime')
            plt.scatter(x='proponeface', y='dprime', data=mldf, color=colorscheme[spind])

            # Store across experiments
            mldf['spind']=spind
            allmldf = allmldf.append(mldf)
            allpred.append(pred)

    fig = plt.figure("ROC")
    plt.xlabel('False alarm rate')
    plt.ylabel('Hit rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    fig.savefig(os.path.join(outpth, 'ROC.pdf'), format='pdf')

    fig = plt.figure("boundingbox")
    plt.xlim([-0.3, 1.3])
    plt.ylim([-0.3, 1.3])
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, alpha=0.1))
    fig.savefig(os.path.join(outpth, 'boundingbox.pdf'), format='pdf')

    fig = plt.figure('Confidence-dprime')
    sns.regplot(x='Confidence', y='dprime', marker="", line_kws={'color': '0.5'}, data=allmldf, dropna=True)
    plt.xlabel('Confidence of Face Detection')
    plt.ylabel('d-prime')
    fig.savefig(os.path.join(outpth, 'Confidence-dprime.pdf'), format='pdf')

    fig = plt.figure('QualitySharpness-dprime')
    sns.regplot(x='QualitySharpness', y='dprime', marker="", line_kws={'color': '0.5'}, data=allmldf, dropna=True)
    plt.xlabel('Quality - Sharpness')
    plt.ylabel('d-prime')
    fig.savefig(os.path.join(outpth, 'QualitySharpness-dprime.pdf'), format='pdf')

    fig = plt.figure('QualityBrightness-dprime')
    sns.regplot(x='QualityBrightness', y='dprime', marker="", line_kws={'color': '0.5'}, data=allmldf, dropna=True)
    plt.xlabel('Quality - Brightness')
    plt.ylabel('d-prime')
    fig.savefig(os.path.join(outpth, 'QualityBrightness-dprime.pdf'), format='pdf')

    fig = plt.figure('proponeface-dprime')
    sns.regplot(x='proponeface', y='dprime', marker="", line_kws={'color': '0.5'}, data=allmldf, dropna=True)
    plt.xlabel('PropOneFace')
    plt.ylabel('d-prime')
    fig.savefig(os.path.join(outpth, 'proponeface-dprime.pdf'), format='pdf')

    plt.show()

    return {'allpred': allpred, 'allmldf': allmldf}


def inter_rater_reliability(bucket, experimentfilter, mldf, pred, outpth, QualityBrightnessThreshold=None, color=None):
    """
    Calculates Cohen's Kappa across raters
    Rhodri Cusack Trinity College Dublin 2018-03-11 rhodri@cusacklab.org
    :param bucket: s3 bucket for auto coding to load
    :param experimentfilter: experiment to work on (path in s3 for coded results)
    :return:
    """

    if not color:
        color = 'r'

    s3 = boto3.resource('s3')
    s3bucket = s3.Bucket(bucket)

    # Download each subject and add to dataframe
    df = pd.DataFrame({'man_man_kappa': [], 'man_auto_kappa': []})
    for subjind, codobj in enumerate(s3bucket.objects.filter(Prefix=experimentfilter)):
        # This is where the output will be written
        outkey = os.path.join('coding_summary', experimentfilter, 'summary.pickle')

        man_man_kappa = []
        man_auto_kappa = []
        fn = s3tools.getpath({'S3Bucket': bucket, 'S3ObjectName': codobj.key})
        mldf_thissubj = mldf[mldf.index.str.match(codobj.key)]
        with open(fn, 'rb') as f:
            if QualityBrightnessThreshold is None or (
                    mldf_thissubj['QualityBrightness'] >= QualityBrightnessThreshold).any():
                obj = pickle.load(f)
                m = [[oneperson['code'] for oneperson in c['mancod_allraters'] if not oneperson['code'] is None] for c
                     in obj['coding']]
                m = np.array(m)
                a = pred[subjind]
                q = np.concatenate((np.array(a).reshape((-1, 1)), m), axis=1)
                for pair in combinations(range(np.size(m, 1)), 2):
                    man_man_kappa.append(cohen_kappa_score(m[:, pair[0]], m[:, pair[1]]))
                for manind in range(np.size(m, 1)):
                    man_auto_kappa.append(cohen_kappa_score(m[:, manind], a))
                df = df.append(
                    pd.DataFrame({'man_man_kappa': np.mean(man_man_kappa), 'man_auto_kappa': np.mean(man_auto_kappa)},
                                 index=[codobj.key]))


    with open(os.path.join(outpth, 'kappa.txt'), 'a') as fout:
        print(df, file=fout)
    return df


if __name__ == '__main__':
    bucket = 'infantrekognition'
    usemedianforcentering=True

    for task in ['looking-or-not', 'preference-looking']:

        prs = {}

        if task == 'looking-or-not':
            colorscheme = ['tab:blue', 'tab:orange', 'tab:green']
            # For figures and summaries
            outpth = '/imaging/rcusack/Dropbox/python/aws_video/figures'
            if usemedianforcentering:
                outpth=outpth+'_median'
            if not os.path.exists(outpth):
                os.mkdir(outpth)

            experimentfilters = ['osf/dqmcv/osfstorage/oneshot_coded', 'osf/tqgkc/osfstorage/oneshot_coded', 'raw']

            # Tell Rekognition to process them
            # for experimentfilter in experimentfilters:
            #    run_video_rekognition(bucket, experimentfilter)

            # Need one of the following
            # (1)
            #   Pull responses from SQS queue, get rekognition response, extract details, annotate video
            # process_sqs_responses(bucket,'anothersqs',doevenifdone=False)
            # (2)
            #   Reprocess videos, skippnig extraction of allfaces from rekognition
            #            for experimentfilter in experimentfilters:
            #                reprocess_video(bucket,experimentfilter)

            # For debug
            # compmsg={'JobId': "725251f6d446b05f0fc3796d494446dcc5f078a0e047b9fe57dfae9ab18d3279",
            #         'Video':{'S3ObjectName': 'osf/dqmcv/osfstorage/oneshot_coded/oneshot_1181_child0_~h~g~4_free.mp4', 'S3Bucket': 'infantrekognition'}}
            # process_rekognition_video(bucket,compmsg,True)
            # reprocess_video(bucket,'raw/3S0TNUHWKTHUAZGRSNT5QY5B9N78D8')
            # post_rekognition_summary(bucket, 'coding/raw/3S0TNUHWKTHUAZGRSNT5QY5B9N78D8')

            # Aggregate manual and auto-coding values for each subject
            #for ind, experimentfilter in enumerate(experimentfilters):
            #    prs[experimentfilter] = post_rekognition_summary(bucket, 'coding/' + experimentfilter)

            # Run leave-one-out machine learning
            mlres = run_machine_learning(bucket, ['coding_summary/coding/' + experimentfilter for experimentfilter in
                                                  experimentfilters],
                                         outpth, possible_codes=[1, 2], colorscheme=colorscheme, usemedianforcentering=usemedianforcentering)

        elif task == 'preference-looking':
            colorscheme = ['tab:red', 'tab:purple']

            # For figures and summaries
            outpth = '/imaging/rcusack/Dropbox/python/aws_video/figures_preflooking'
            if usemedianforcentering:
                outpth=outpth+'_median'
            if not os.path.exists(outpth):
                os.mkdir(outpth)

            experimentfilters = ['osf/dqmcv/osfstorage/novelverbs', 'osf/tqgkc/osfstorage/novelverbs']

            # Tell Rekognition to process them
            #        for experimentfilter in experimentfilters:
            #            run_video_rekognition(bucket, experimentfilter)

            #   Pull responses from SQS queue, get rekognition response, extract details, annotate video
            #        process_sqs_responses(bucket,'anothersqs',doevenifdone=False)

            #        #   Reprocess videos, skippnig extraction of allfaces from rekognition
            #            for experimentfilter in experimentfilters:
            #                reprocess_video(bucket,experimentfilter)

            # Aggregate manual and auto-coding values for each subject
            #for experimentfilter in experimentfilters:
            #    prs[experimentfilter] = post_rekognition_summary(bucket, 'coding/' + experimentfilter)

            # Run leave-one-out machine learning
            mlres = run_machine_learning(bucket, ['coding_summary/coding/' + experimentfilter for experimentfilter in
                                                  experimentfilters], outpth, possible_codes=[-1, 1],
            colorscheme=colorscheme, usemedianforcentering=usemedianforcentering)

        # Histograms of age ranges of faces detected
        if prs:
            for agelimit in ['low']:
                fig_ar = plt.figure("agerange%s" % agelimit )
                plt.hist([prs[x]["agerange%s" % agelimit] for x in prs], stacked=True, color=colorscheme,
                         bins=np.linspace(0, 79.5, 80))
                fig_ar.savefig(os.path.join(outpth, "agerange%s.pdf"%agelimit), format='pdf')

        if mlres:
            # I'm sure there's a more elegant way to do this grouping
            groups=mlres['allmldf']['proponeface'].groupby(mlres['allmldf']['spind'])
            fig = plt.figure("proponeface")
            plt.hist([list(x[1]) for x in groups], stacked=True, color=colorscheme, bins=np.linspace(0, 1, 10))
            plt.xlabel('Proportion one face')
            plt.ylabel('Number of videos')
            fig.savefig(os.path.join(outpth, 'proponeface_stacked.pdf'), format='pdf')

        # Common across experiments
        # Summarise inter-rater reliability
        irr = {}
        with open(os.path.join(outpth, 'kappa.txt'), 'w') as fout:
            print("Inter rater reliability", file=fout)
        for ind, experimentfilter in enumerate(experimentfilters):
            irr[experimentfilter] = inter_rater_reliability(bucket, 'coding/' + experimentfilter, mlres['allmldf'],
                                                            mlres['allpred'][ind],
                                                            outpth, color=colorscheme[ind])
        # Inter-rater reliability figures
        fig = plt.figure('kappa')
        plt.subplot(211)
        plt.hist([thing['man_man_kappa'].dropna() for key, thing in irr.items()], stacked=True, color=colorscheme, bins=np.linspace(-1, 1, 20))
        plt.xlim([-1, 1])
        plt.xlabel('')
        plt.ylabel('Human-Human')

        plt.subplot(212)
        plt.hist([thing['man_auto_kappa'].dropna() for key, thing in irr.items()], stacked=True, color=colorscheme, bins=np.linspace(-1, 1, 20))
        plt.xlim([-1, 1])
        plt.xlabel('Inter-rater reliability (Cohen''s kappa)')
        plt.ylabel('Human-classifier')

        fig.savefig(os.path.join(outpth, 'kappa.pdf' ), format='pdf')


        # Make table of irr
        with open(os.path.join(outpth, 'irr.txt'), 'w') as f:
            for key, item in irr.items():
                print("%s" % key, file=f, end="\t")
                mns = item.mean()
                stds = item.std()
                for colkey, colitem in mns.items():
                    print("%s\t%f +/- %f" % (colkey, colitem, stds[colkey]), file=f, end="\t")
                print("", file=f)
        mlres['allmldf'].to_csv(os.path.join(outpth, 'allmldf.csv'))
        with open(os.path.join(outpth, 'allmldf.pickle'), 'wb') as f:
            pickle.dump(mlres['allmldf'], f)
    plt.show()
