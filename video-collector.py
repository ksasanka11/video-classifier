#!/usr/bin/python
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse
from pytube import YouTube
import pandas as pd
import speech_recognition as sr

load_dotenv()

# constants to search youtube using googleapiclient
DEVELOPER_KEY = os.environ.get('API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

DATA_HEADERS = ['title', 'id', 'category']
YOUTUBE_LINK = 'https://www.youtube.com/watch?v='
AUDIO_DOWNLOAD_PATH = './audio/'

# initialize recognizer
recognizer = sr.Recognizer()

try:
	videos_data = pd.read_csv('data.csv')
	videos = videos_data.to_numpy().tolist()
except Exception as e:
	# global videos lists
	# videos_data = pd.DataFrame([], columns=DATA_HEADERS)
	# videos_data.to_csv('data.csv', mode='w', index=False)
	videos = []

# download audio from videos
def download_audio(unique_id):
	# for video_item in videos:
	try:
		# unique_id = video_item[1]
		# print(unique_id)
		video = YouTube(YOUTUBE_LINK+unique_id)
		# filtering the audio. File extension can be mp4/webm
		# You can see all the available streams by print(video.streams)
		audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
		audio.download(output_path=AUDIO_DOWNLOAD_PATH, filename=unique_id+'.mp4')
	except Exception as e:
		print("Connection Error")
		print(e)

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
			video_details = [search_result['snippet']['title'], search_result['id']['videoId'], options.q]
			videos.append(video_details)
			print('Downloading '+video_details[0]+'......')
			download_audio(video_details[1])
			print('Downloaded')


# search for youtube videos
# categories = ['computer science', 'biology', 'environmental studies']
categories = ['computer science']
if len(videos) <= 0:
	for category in categories:
		parser = argparse.ArgumentParser()
		parser.add_argument('--q', help='Search term', default=category)
		parser.add_argument('--max-results', help='Max results', default=1)
		args = parser.parse_args()

		try:
			youtube_search(args)
		except HttpError as e:
			print('An HTTP error '+e.resp.status+' occurred:\n' + e.content)
	
	videos_df = pd.DataFrame(videos, columns=DATA_HEADERS)
	videos_df.to_csv('data.csv', mode='a', index=False)