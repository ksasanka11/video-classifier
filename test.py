from pydub import AudioSegment
import speech_recognition as sr


# mp4_audio = AudioSegment.from_file("./audio/Tzl0ELY_TiM.mp4", format="mp4")
# file_handle = mp4_audio.export("./test.wav", format="wav", parameters=['-ac', '1'])

recognizer = sr.Recognizer()
# with sr.AudioFile('./test.wav') as source:
#     audio_text = recognizer.listen(source)
#     try:
#         # using google speech recognition
#         text = recognizer.recognize_google(audio_text)
#         print('Converting audio transcripts into text ...')
#         print(text)
        
#     except Exception as e:
#             print(e)


from youtube_transcript_api import YouTubeTranscriptApi
  
# assigning srt variable with the list
# of dictionaries obtained by the get_transcript() function
srt = YouTubeTranscriptApi.get_transcript("Tzl0ELY_TiM")
  
# prints the result
text = []
for _ in srt:
    text.append(_['text'])

print(' '.join(text))