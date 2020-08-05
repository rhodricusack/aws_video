import boto3
import hashlib
import pickle

''' 
Runs AWS face rekognition on all of the files in a directory
'''

def run_video_rekognition(bucket,prefix):

    # List all objects in the bucket
    s3=boto3.client('s3')
    allfiles=s3.list_objects(Bucket=bucket,Prefix=prefix)

    # For testing just one file
    #allfiles['Contents']=[allfiles['Contents'][2]]

    # Create mediaconvert object
    rekognition=boto3.client('rekognition')

    # For each file create transcode job
    for myobj in allfiles['Contents']:
        # Only process objects with .mp4 suffix
        if myobj['Key'][-4:]=='.mp4':
            print(myobj['Key'])
            filename=myobj['Key']
            response=rekognition.start_face_detection(
                Video={'S3Object': {
                    'Bucket':bucket, 'Name':filename}
                },
                ClientRequestToken=hashlib.md5(str('s3://' + bucket+'/'+filename).encode()).hexdigest(),
                FaceAttributes='ALL',
                NotificationChannel={
                    'SNSTopicArn': 'arn:aws:sns:eu-west-1:807820536621:AmazonRekognition_cusack_infant',
                    'RoleArn':'arn:aws:iam::807820536621:role/cusack_infant_rekognition'
                }
            )
            print(response['JobId'])
            print(response)

