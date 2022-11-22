#!/usr/bin/python
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse
from pytube import YouTube
import pandas as pd

load_dotenv()

# constants to search youtube using googleapiclient
DEVELOPER_KEY = os.environ.get('API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# videos_data = pd.read_csv('video-data.csv')

# global videos lists
videos = []

YOUTUBE_LINK = 'https://www.youtube.com/watch?v='
AUDIO_DOWNLOAD_PATH = './audio/'

def youtube_search(options):
	youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    # Call the search.list method to retrieve results matching the specified
    # query term.
	search_response = youtube.search().list(q=options.q, type='video', videoDuration='medium',
            part='id,snippet', maxResults=options.max_results).execute()

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
	for search_result in search_response.get('items', []):
		if search_result['id']['kind'] == 'youtube#video':
			video_details = (search_result['snippet']['title'], search_result['id']['videoId'], options.q)
			videos.append(video_details)

# download audio from videos
def download_audio():
	for video_item in videos:
		try:
			unique_id = video_item[1]
			video = YouTube(YOUTUBE_LINK+unique_id)
			# filtering the audio. File extension can be mp4/webm
			# You can see all the available streams by print(video.streams)
			audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
			audio.download(output_path=AUDIO_DOWNLOAD_PATH+unique_id)
		except Exception as e:
			print("Connection Error")
			print(e)


# search for youtube videos
# categories = ['computer science', 'biology', 'environmental studies']
categories = ['computer science']

for category in categories:
	parser = argparse.ArgumentParser()
	parser.add_argument('--q', help='Search term', default=category)
	parser.add_argument('--max-results', help='Max results', default=1)
	args = parser.parse_args()

	try:
		youtube_search(args)
	except HttpError as e:
		print('An HTTP error '+e.resp.status+' occurred:\n' + e.content)

# download audio files from each of the videos
download_audio()