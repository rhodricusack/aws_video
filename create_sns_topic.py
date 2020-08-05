import boto3

sns=boto3.client('sns')
print sns.create_topic(Name='AmazonRekognition_cusack_infant')

response = sns.publish(
    TopicArn='arn:aws:sns:eu-west-1:807820536621:AmazonRekognition_cusack_infant',
    Message='hi test message'
)
print("Response: {}".format(response))

