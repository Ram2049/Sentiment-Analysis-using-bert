from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import re 
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import numpy as np
import re

tokenizer = AutoTokenizer.from_pretrained('D:\Project\MINI PROJECT\Sentiment analysis application\model')
model = AutoModelForSequenceClassification.from_pretrained('D:\Project\MINI PROJECT\Sentiment analysis application\model')



def predict(inputtext):
    tokens = tokenizer.encode(inputtext,return_tensors='pt')
    results = model(tokens)
    predictions_score=int(torch.argmax(results.logits))+1
    return predictions_score


def get_video_id(youtube_link):
    """
    Extracts the video ID from a YouTube URL.
    
    :param youtube_link: str, YouTube video URL
    :return: str or None, extracted video ID or None if invalid
    """
    try:
        # Regular expression to match YouTube video IDs
        patterns = [
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)",  # Standard YouTube URL
            r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)",  # Shortened youtu.be URL
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)"  # Embedded video URL
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_link)
            if match:
                return match.group(1)  # Extract video ID
        
        print("Unsupported YouTube link format.")
        return None  # If no match found, return None

    except Exception as e:
        print(f"Error extracting video ID: {e}")
        return None

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

    df.drop(columns=['author', 'published_at', 'updated_at', 'like_count'],inplace=True)

    df['sentiment']=df['text'].apply(lambda x : predict(x[:100]))
    
    emoji_mapping = {
    1: '😡',  # Sad face
    2: '🙄',  # Neutral face
    3: '😐',  # Smiling face
    4: '😁',  # Grinning face
    5: '🤩'   # Star-struck
    }

    df['Emoji'] = df['sentiment'].map(emoji_mapping)

    return df
 
