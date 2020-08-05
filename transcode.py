import boto3
import hashlib
import pickle

''' 
Uses AWS media convert to transcode all .flv files in an S3 bucket into .mp4 files
'''
# You set these. To get endpoint, run
#  mediaconvert=boto3.client('mediaconvert')
#  mediaconvert.describe_endpoints()

bucket="infantrekognition"
endpoint_url='https://fdeup1ey.mediaconvert.eu-west-1.amazonaws.com'

# List all objects in the bucket
s3=boto3.client('s3')
allfiles=s3.list_objects(Bucket=bucket)

# Create mediaconvert object
mediaconvert=boto3.client('mediaconvert',endpoint_url=endpoint_url)

# For each file create transcode job
for myobj in allfiles['Contents']:
    # Only process objects with .flv suffix
    ext=myobj['Key'][-4:]
    if ext=='.flv':
        print "File %s"%myobj['Key']
        filename="s3://" + bucket +"/"+myobj['Key']

        settings={
            "OutputGroups": [
              {
                "Name": "File Group",
                "Outputs": [
                  {
                    "ContainerSettings": {
                      "Container": "MP4",
                      "Mp4Settings": {
                        "CslgAtom": "INCLUDE",
                        "FreeSpaceBox": "EXCLUDE",
                        "MoovPlacement": "PROGRESSIVE_DOWNLOAD"
                      }
                    },
                    "VideoDescription": {
                      "ScalingBehavior": "DEFAULT",
                      "TimecodeInsertion": "DISABLED",
                      "AntiAlias": "ENABLED",
                      "Sharpness": 50,
                      "CodecSettings": {
                        "Codec": "H_264",
                        "H264Settings": {
                          "InterlaceMode": "PROGRESSIVE",
                          "NumberReferenceFrames": 3,
                          "Syntax": "DEFAULT",
                          "Softness": 0,
                          "GopClosedCadence": 1,
                          "GopSize": 90,
                          "Slices": 1,
                          "GopBReference": "DISABLED",
                          "SlowPal": "DISABLED",
                          "SpatialAdaptiveQuantization": "ENABLED",
                          "TemporalAdaptiveQuantization": "ENABLED",
                          "FlickerAdaptiveQuantization": "DISABLED",
                          "EntropyEncoding": "CABAC",
                          "Bitrate": 5000000,
                          "FramerateControl": "INITIALIZE_FROM_SOURCE",
                          "RateControlMode": "CBR",
                          "CodecProfile": "MAIN",
                          "Telecine": "NONE",
                          "MinIInterval": 0,
                          "AdaptiveQuantization": "HIGH",
                          "CodecLevel": "AUTO",
                          "FieldEncoding": "PAFF",
                          "SceneChangeDetect": "ENABLED",
                          "QualityTuningLevel": "SINGLE_PASS",
                          "FramerateConversionAlgorithm": "DUPLICATE_DROP",
                          "UnregisteredSeiTimecode": "DISABLED",
                          "GopSizeUnits": "FRAMES",
                          "ParControl": "INITIALIZE_FROM_SOURCE",
                          "NumberBFramesBetweenReferenceFrames": 2,
                          "RepeatPps": "DISABLED"
                        }
                      },
                      "AfdSignaling": "NONE",
                      "DropFrameTimecode": "ENABLED",
                      "RespondToAfd": "NONE",
                      "ColorMetadata": "INSERT"
                    },
                    "Extension": ".mp4"
                  }
                ],
                "OutputGroupSettings": {
                  "Type": "FILE_GROUP_SETTINGS",
                  "FileGroupSettings": {
                    "Destination": "s3://infantrekognition/"
                  }
                }
              }
            ],
            "AdAvailOffset": 0,
            "Inputs": [
              {
                "VideoSelector": {
                  "ColorSpace": "FOLLOW"
                },
                "FilterEnable": "AUTO",
                "PsiControl": "USE_PSI",
                "FilterStrength": 0,
                "DeblockFilter": "DISABLED",
                "DenoiseFilter": "DISABLED",
                "TimecodeSource": "EMBEDDED",
                "FileInput": filename
              }
            ]
          }

        response = mediaconvert.create_job(
          ClientRequestToken=hashlib.md5(pickle.dumps(settings)).hexdigest(),
          Queue= "arn:aws:mediaconvert:eu-west-1:807820536621:queues/Default",
          UserMetadata= {},
          Role="arn:aws:iam::807820536621:role/Cusack_MediaConvert",
          Settings= settings
        )
        print response
    else:
        print "Ignoring %s"%myobj['Key']


