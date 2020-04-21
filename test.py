import os
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import googleapiclient.discovery
import youtube_dl


SCRAPE_ROOT = "scrape"

API_SERVICE = "youtube"
API_VERSION = "v3"
API_KEY = 'AIzaSyCQ0PV7iXCRn3xAToald-KvXEdp7q6YSSI'


def sanitize(string):
    """
    Replace all substrings of non-alphanumeric characters with underscore
    """
    return re.sub('[^0-9a-zA-Z]+', '_', string)


def get_album_elements(driver):
    """
    Assumes that the driver is currently on the artist page
    Returns a list of elements corresponding to albums/singles
    """
    relevant_headers = {"Albums", "Singles"}
    
    carousels = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "ytmusic-carousel-shelf-renderer")))
    carousels = [carousel for carousel in carousels if carousel.find_element_by_class_name("header").text in relevant_headers]
    return [item for carousel in carousels for item in carousel.find_elements_by_tag_name("ytmusic-two-row-item-renderer")]


def get_album_playlist_ids(driver):
    """
    Assumes that the driver is already on the artist page
    """
    playlist_ids = {}
    
    items = get_album_elements(driver)
    for i in range(len(items)):
        if i > 0:
            items = get_album_elements(driver)
        
        item = items[i]
        item_title = sanitize(item.find_element_by_class_name("details").find_element_by_class_name("title").text)
        item.click()

        WebDriverWait(driver, 10).until(EC.url_contains("playlist"))
        playlist_ids[item_title] = driver.current_url[driver.current_url.index('=') + 1:]
        driver.back()
        
    return playlist_ids
            

def get_channel_info(search_string):
    try:
        driver = webdriver.Chrome()
        #driver.get("https://music.youtube.com")
        driver.get(f"https://music.youtube.com/search?q={search_string}")
    
        #search_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "search-container")))
        #search_icon = WebDriverWait(search_container, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "search-icon")))
        #search_icon.click()
        #search_input = WebDriverWait(search_container, 10).until(EC.element_to_be_clickable((By.ID, "input")))
        #search_input.send_keys(search_string, Keys.ENTER)
    
        top_result = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "ytmusic-shelf-renderer")))[0]
        top_result.click()
        
        WebDriverWait(driver, 10).until(EC.url_contains("channel"))
        channel_id = driver.current_url.split('/')[-1]
    
        return channel_id, get_album_playlist_ids(driver)
    finally:
        driver.quit()
    
    return "", {}


def get_video_ids_in_playlist(playlist_id):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = googleapiclient.discovery.build(API_SERVICE, API_VERSION, developerKey=API_KEY)
    
    video_ids = []
    pageToken = None
    while True:
        request_kwargs = {'part': 'snippet', 'playlistId': playlist_id, 'maxResults': 50}
        if pageToken:
            request_kwargs['pageToken'] = pageToken

        request = youtube.playlistItems().list(**request_kwargs)
        response = request.execute()
        video_ids.extend([item['snippet']['resourceId']['videoId'] for item in response['items']])
        
        if 'nextPageToken' in response:
            pageToken = response['nextPageToken']
        else:
            break
    
    return video_ids


def download_as_wav(video_ids, output_dir=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }]
    }
    if output_dir:
        ydl_opts['outtmpl'] = f'{test_dir}/%(title)s.%(ext)s'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={video_id}' for video_id in video_ids])


def main():
    channel_id, playlist_ids = get_channel_info("itzy")
    
    for playlist_name, playlist_id in playlist_ids:
        video_ids = get_video_ids_in_playlist(playlist_id)
        download_as_wav(video_ids, playlist_name)

        
if __name__ == '__main__':
    main()
