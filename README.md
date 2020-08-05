# 2020-08-05
## rekognition_all_stages.py
Main entry point to run rekognition.

# 2018-02-26 
## run_video_rekognition.py
    Picks up mp4 files from S3 and runs rekognition on them
## process_rekognition_video.py
    Retrieves completion messages from SQS and matches behavioural data with automatically coded data
    Saves results in pickle files in s3://infantrekognition/coding
    Creates annotated videos, which it saves in s3://infantrekognition/annotated
## post_rekognition_ml.py
    Loads coding pickle files and extracts creates Pandas dataframe containing columns that are likely to be useful for ML

# 2018-03-12 
Now all in one file, rekognition_all_stages.py
