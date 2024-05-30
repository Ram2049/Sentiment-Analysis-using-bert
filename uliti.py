from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import re 
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import numpy as np

autotokenizer_path = 'D:\Project\MINI PROJECT\Sentiment analysis application\model'
pretrainedmodel = 'D:\Project\MINI PROJECT\Sentiment analysis application\model'

tokenizer = AutoTokenizer.from_pretrained(autotokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrainedmodel)



def predict(inputtext):
    tokens = tokenizer.encode(inputtext,return_tensors='pt')
    results = model(tokens)
    predictions_score=int(torch.argmax(results.logits))+1
    return predictions_score

def video_id(youtube_link):
    video_id = None

    if "youtube.com/watch?v=" in youtube_link:
        video_id = youtube_link.split("youtube.com/watch?v=")[1]
    elif "youtu.be/" in youtube_link:
        video_id = youtube_link.split("youtu.be/")[1]
    elif "youtube.com/embed/" in youtube_link:
        video_id = youtube_link.split("youtube.com/embed/")[1]
    else:
        print("Unsupported YouTube link format.")

    if video_id:
        # Extracting video ID from the URL (if there are additional parameters)
        video_id = video_id.split("&")[0]
    else:
        print("Failed to extract video ID.")

    return video_id

def yt_predict(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBvOcknQzZ4xTdyyauVQCq2uejJcF_-Hzw"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['updatedAt'],
            comment['likeCount'],
            comment['textDisplay']
        ])

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

    df.drop(columns=['author', 'published_at', 'updated_at', 'like_count'])

    df['sentiment']=df['text'].apply(lambda x : predict(x[:100]))

    return df
 
