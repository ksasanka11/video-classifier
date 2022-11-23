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
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm

load_dotenv()

# constants to search youtube using googleapiclient
DEVELOPER_KEY = os.environ.get('API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

DATA_HEADERS = ['title', 'id', 'category', 'transcript']
YOUTUBE_LINK = 'https://www.youtube.com/watch?v='
AUDIO_DOWNLOAD_PATH = './audio/'

NUMBER_OF_VIDEOS_PER_CATEGORY = 100
categories = ['computer science', 'biology', 'environmental studies']
TRANSCRIPT_LANGS = ['en', 'en-AU', 'en-BZ', 'en-CA', 'en-IE', 'en-JM', 'en-NZ', 'en-ZA', 'en-TT', 'en-GB', 'en-US']
# categories = ['computer science']

# read/create dataset
try:
	videos_data = pd.read_csv('data.csv')
	videos = videos_data.to_numpy().tolist()
except Exception as e:
	videos = []

# download audio from videos
def download_audio(unique_id):
	try:
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

    # Call the search.list method to retrieve results matching the specified query term.
	search_response = youtube.search().list(q=options.q, type='video', videoDuration='short', videoCaption='closedCaption',
            part='id,snippet', maxResults=options.max_results).execute()

    # Add each result to the appropriate list, and then display the lists of matching videos, channels, and playlists.
	for search_result in tqdm(search_response.get('items', [])):
		if search_result['id']['kind'] == 'youtube#video':
			video_details = [search_result['snippet']['title'], search_result['id']['videoId'], options.q]
			tqdm.write(video_details[0])
			srt = YouTubeTranscriptApi.get_transcript(video_details[1], languages=TRANSCRIPT_LANGS)
			# prints the result
			text = []
			for _ in srt:
				text.append(_['text'])
			video_details.append(' '.join(text).replace('\n', ' '))
			videos.append(video_details)

# search for youtube videos
if len(videos) <= 0:
	categories_progress_bar = tqdm(categories)
	for category in categories_progress_bar:
		parser = argparse.ArgumentParser()
		parser.add_argument('--q', help='Search term', default=category)
		parser.add_argument('--max-results', help='Max results', default=NUMBER_OF_VIDEOS_PER_CATEGORY)
		args = parser.parse_args()

		try:
			youtube_search(args)
		except HttpError as e:
			print('An HTTP error '+e.resp.status+' occurred:\n' + e.content)
	
	videos_df = pd.DataFrame(videos, columns=DATA_HEADERS)
	videos_df.to_csv('data.csv', mode='a', index=False)