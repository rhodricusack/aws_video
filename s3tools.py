import boto3,botocore
from pathlib import Path
import os

def getcacheinpath(key):
    # Given an S3 key name, create a local filename to cache it
    return  os.path.join(Path.home(), ".s3cache/", key)

def getcacheoutpath(key):
    # Given a proposed S3 key name, create a local filename to cache it
    outfn=os.path.join(Path.home(), ".s3cache-out", key)
    # Make directory if necessary
    dirname = os.path.dirname(outfn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return outfn

def getpath(pth):
    # Given a local path or S3 path, return a local pathname of the file, downloaded if necessary
    iss3 = False

    if type(pth) is dict:
        # S3 paths may be provided as dict with S3Bucket and S3ObjectName keys
        bucket = pth['S3Bucket']
        key = pth['S3ObjectName']
        iss3 = True
    elif pth.startswith('s3://'):
        # S3 paths can be given in as a string in the form s3://bucket/path/to/object
        pthparts = pth.split('/')
        bucket = pthparts[2];
        key = '/'.join(pthparts[3:])
        iss3 = True

    # Download from S3 if we found one of these
    if iss3:
        s3 = boto3.resource('s3')
        fname = getcacheinpath( key)

        # Make directory if necessary
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        try:
            if os.path.exists(fname):
                os.remove(fname)
            print("Downloading from bucket %s key %s" % (bucket, key))
            s3.Bucket(bucket).download_file(key, fname)
            print("Done")
            return fname
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    else:
        return pth
