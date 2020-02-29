import json
import boto3
from pre_processing.pre_processing import PreProcessor
import datetime

my_pre_processor = PreProcessor(padding_size=40, max_dictionary_size=100000)
sage_maker_client = boto3.client("runtime.sagemaker")
s3_client = boto3.client("s3")

def lambda_handler(event, context):

    request_time = datetime.datetime.today()
    
    tweet = event["tweet"]

    begin_pre_process_time = datetime.datetime.now()

    features = my_pre_processor.pre_process_text(tweet)

    pre_process_time = datetime.datetime.now() - begin_pre_process_time

    model_payload={
        'features_input': features
    }
    
    begin_model_inference_time = datetime.datetime.now()

    model_response = sage_maker_client.invoke_endpoint(
        EndpointName = "sentiment-model-aiops",
        ContentType = "application/json",
        Body = json.dumps(model_payload))

    model_inference_time = datetime.datetime.now() - begin_model_inference_time

    result = json.loads(model_response["Body"].read().decode())

    request_time = datetime.datetime.strftime(request_time,'%y-%m-%d %H:%M:%S')

    response = {
        "Request Time": request_time,
        "Tweet": tweet,
        "Sentiment": None,
        "Probability": result["predictions"][0][0],
        "Pre Processing Time": str(pre_process_time),
        "Model Inference Time":str(model_inference_time)
    }

    if response["Probability"] >= 0.5:
        response["Sentiment"] = "positive"
    else:
        response["Sentiment"] = "negative"

    S3_response = s3_client.put_object(
        Bucket = "assignment5-logs",
        Key = "lambda_sentiment_analysis/payload/"+request_time+".json",
        Body = json.dumps(response,indent=2))

    print("Result:" + json.dumps(response, indent=2))
    
    return {"sentiment":response["Sentiment"]}
