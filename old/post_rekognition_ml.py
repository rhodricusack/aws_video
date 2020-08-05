import os
import boto3
import s3tools
import pickle
import pandas as pd

'''
Aggregates data across subjects from auto and manual coding
Rhodri Cusack Trinity College Dublin 2018-03-11 rhodri@cusacklab.org
'''
bucket='infantrekognition'
# Write annotated video and coding files to s3
s3 = boto3.resource('s3')
s3bucket=s3.Bucket(bucket)

experimentfilter='coding/osf/dqmcv/osfstorage/oneshot_coded'

# Columns for machine learning
pose_columns=['Pitch', 'Yaw', 'Roll']
landmark_columns=[['leftPupil','X'],['rightPupil','X'],['eyeLeft','X'],['eyeRight','X']]

df=[]

# Download each subject and add to dataframe
for codobj in s3bucket.objects.filter(Prefix=experimentfilter):
    fn=s3tools.getpath({'S3Bucket':bucket,'S3ObjectName':codobj.key})
    df.append({'S3Bucket':bucket,'S3ObjectName':codobj.key,'df':pd.DataFrame(columns=pose_columns + [x[0] for x in landmark_columns] + ['mancod'])})
    with open(fn,'rb') as f:
        obj=pickle.load(f)

        # Get the face details
        q = [[x['faces'][y] for y in x['infantind']] for x in obj['coding']] # "faces" contains indices within allfaces. "infantind" contains indices within faces. Get indices of infants among allfaces.
        myfaces=[[obj['allfaces'][z1] for z1 in z0] for z0 in q]             # get actual face data from allfaces

        for ind,item in enumerate(myfaces):
            # Build a row
            row = {}

            # One infant face?
            if len(item)==1:
                # Pose
                for col in pose_columns:
                    row[col]=item[0]['Face']['Pose'][col]
                # Landmarks
                for col in landmark_columns:
                    row[col[0]]=[x[col[1]] for x in item[0]['Face']['Landmarks'] if x['Type']==col[0]][0]
                # Eyes open
                row['EyesOpen']=item[0]['Face']['EyesOpen']['Value'] or item[0]['Face']['EyesOpen']['Confidence'] < 90

            # And mean manual coder
            list=[x['code'] for x in obj['coding'][ind]['mancod_allraters']]
            row['mancod']=float(max(set(list), key=list.count))
            df[-1]['df']=df[-1]['df'].append(pd.DataFrame([row]),ignore_index=True)

    print(df[-1]['df'].describe())

# Save result and upload to S3
key=os.path.join('coding_summary',experimentfilter,'summary.pickle')
fname=s3tools.getcacheoutpath(key)
with open(fname,'wb') as f:
    pickle.dump(df,f)
s3 = boto3.resource('s3')
s3.Bucket(bucket).upload_file(fname,key)

# Done!
print("All done")
