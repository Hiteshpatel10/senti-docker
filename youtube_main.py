import os
from utility.youtube.youtube_comments import get_comments
from utility.youtube.youtube_senti import senti 
from googleapiclient.discovery import build
from utility.youtube.youtube_comments import extract_video_id

def youtubeMain(videoId, uuid):
    api_service_name = "youtube"
    api_version = "v3"
    # api_key = os.environ.get("YOUTUBE_API_KEY") 
    yt_service = build(api_service_name, api_version, developerKey="AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4")

    get_comments(yt_service, videoId, uuid)
    senti(uuid)

# def youtubeMain():

#     api_service_name = "youtube"
#     api_version = "v3"
#     # api_key = os.environ.get("YOUTUBE_API_KEY") 
#     yt_service = build(api_service_name, api_version, developerKey="AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4")
    
#     videoId = extract_video_id(videoId)
#     get_comments(yt_service, videoId, 'youtube')
#     senti('youtube')

# def youtubeMain():

#     video_urls = [
           
#     ]

#     api_service_name = "youtube"
#     api_version = "v3"
#     api_key = os.environ.get("YOUTUBE_API_KEY") 
#     yt_service = build(api_service_name, api_version, developerKey="AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4")

#     for url in video_urls:
#         video_id = extract_video_id(url)
#         get_comments(yt_service, video_id, 'youtube')


if __name__ == "__main__":
    youtubeMain()




     
#         "https://www.youtube.com/watch?v=vpNC1qM_cCE",
#         "https://www.youtube.com/watch?v=q0nHvba_F5U",
#         "https://www.youtube.com/watch?v=kV7YaA0MKm0",
#         "https://www.youtube.com/watch?v=Xkl7U4ZH-9A",
#         "https://www.youtube.com/watch?v=3pjOZ06R9qw",
#         "https://www.youtube.com/watch?v=TCI6PEKBbhQ",
#         "https://www.youtube.com/watch?v=E6BI48sS_lU",
#         "https://www.youtube.com/watch?v=IIAq4rrV7hE",
#         "https://www.youtube.com/watch?v=4od7Vps2AqY",
#         "https://www.youtube.com/watch?v=eGThTHldfS8",
#         "https://www.youtube.com/watch?v=XakRFw5SD2o",