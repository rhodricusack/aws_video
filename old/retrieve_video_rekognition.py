import boto3
import hashlib
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

''' 
Retrieves face_detection results from rekognition 
'''

jobid=u'd397406ef569d7e77a735cdd21b206c05d6364648ca9c1e50bc5663b2aa996ef'

# Create mediaconvert object
rekognition=boto3.client('rekognition')

response=rekognition.get_face_detection(JobId=jobid)
allfaces=response['Faces']
while 'NextToken' in response:
    response=rekognition.get_face_detection(JobId=jobid,NextToken=response['NextToken'])
    allfaces.extend(response['Faces'])
# 3063d461046f3250ebd94192f34853c00e70cb3d714a2ba19e1fa9328753e22d

fig, ax = plt.subplots()
ax.scatter([x['Timestamp'] for x in allfaces])
print response
