import os
import pathlib
import re
import heapq
import pickle
import urllib.parse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import googleapiclient.discovery
import youtube_dl

import youtube_api


CACHE_ROOT = f"{os.path.dirname(os.path.realpath(__file__))}/cache"
DATA_ROOT = f"{os.path.dirname(os.path.realpath(__file__))}/data"

API_SERVICE = youtube_api.get_service_name()
API_VERSION = youtube_api.get_version()
API_KEY = youtube_api.get_key()


class ArtistQueue:

    def __init__(self, seed, cutoff=0):
        self.__seed = seed
        self.__cutoff = -1 * cutoff
    
        try:
            self.__pqueue, self.__seen = self.__read_cache()
        except FileNotFoundError:
            self.__pqueue = [(0, seed, None, 0)]
            self.__seen = set()
        
        # The scale factor for searching deeper in the artist recommmendation graph
        # in the traversal algorithm, used for deciding which artists to place first
        # in the search queue
        self.__gamma = 0.8

    def __get_cache_filename(self):
        return f'{self.__seed}-{self.__cutoff}.queue'

    def __write_cache(self):
        pathlib.Path(CACHE_ROOT).mkdir(exist_ok=True)
        with open(f'{CACHE_ROOT}/{self.__get_cache_filename()}', 'wb') as f:
            pickle.dump(self.__pqueue, f)
            pickle.dump(self.__seen, f)
        
    def __read_cache(self):
        pathlib.Path(CACHE_ROOT).mkdir(exist_ok=True)
        with open(f'{CACHE_ROOT}/{self.__get_cache_filename()}', 'rb') as f:
            return pickle.load(f), pickle.load(f)

    @staticmethod
    def __sanitize(string):
        """
        Remove all non-ascii characters
        """
        return re.sub(r'[^\x00-\x7F]+', '', string)
        
    @staticmethod
    def __get_num_subs_from_text(string):
        """
        From a string of the form "XXX subscribers", return the number of subscribers
        Can be in any of the following forms:
            - 43 subscribers
            - 65K subscribers
            - 4.5M subscribers
        """
        num = string.split()[0]
        if num[-1].lower() == 'k':
            return float(num[:-1]) * 1000
        elif num[-1].lower() == 'm':
            return float(num[:-1]) * 1000000
        elif num[-1].lower() == 'b':
            return float(num[:-1]) * 1000000000
        else:
            return int(num)

    @staticmethod
    def __get_artist_url(artist):
        """
        Get the youtube music page of the artist
        """
        try:
            driver = webdriver.Chrome()
            
            search_query = urllib.parse.quote_plus(artist)
            driver.get(f"https://music.youtube.com/search?q={search_query}")
            
            top_result = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "ytmusic-shelf-renderer")))[0]
            top_result.click()
            
            WebDriverWait(driver, 10).until(EC.url_contains("channel"))
            return driver.current_url
        finally:
            driver.close()
            
    def __get_recommended_artists(self, artist_page, trim=None):
        """
        Given an artist or a page, return a list containing recommended
        artists in the form of (name, youtube_music_url, num_subs) tuples
        """
        try:
            driver = webdriver.Chrome()
            driver.get(artist_page)
    
            carousels = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "ytmusic-carousel-shelf-renderer")))
            carousel = [carousel for carousel in carousels if "Fans" in carousel.find_element_by_class_name("header").text][0]
        
            recommendations = []
            for count, item in enumerate(carousel.find_elements_by_tag_name("ytmusic-two-row-item-renderer")):
                if trim and count >= trim:
                    break
                name = ArtistQueue.__sanitize(item.find_element_by_class_name("details").find_element_by_class_name("title").text)
                url = item.find_element_by_tag_name("a").get_attribute("href")
                num_subs = ArtistQueue.__get_num_subs_from_text(item.find_element_by_class_name("details").find_element_by_class_name("subtitle").text)
            
                recommendations.append((name, url, num_subs))
            
            return recommendations
        except Exception as e:
            return []
        finally:
            driver.close()
    
    def get_next(self):
        """
        Get the name and youtube music page of the next artist in the search queue.
        Internally adds more artists to the queue
        """
        while True:
            next_priority, next_name, next_url, next_depth = heapq.heappop(self.__pqueue)
            if next_name not in self.__seen:
                break
        self.__seen.add(next_name)
        
        if not next_url:
            next_url = ArtistQueue.__get_artist_url(next_name)
        
        recommendations = self.__get_recommended_artists(next_url)
        for name, url, num_subs in recommendations:
            priority = -1 * num_subs * (self.__gamma ** (next_depth + 1))
            if priority <= self.__cutoff:
                heapq.heappush(self.__pqueue, (priority, name, url, next_depth + 1))
        
        self.__write_cache()
        return next_name, next_url
            

class ArtistMetaInfo:

    __ALBUM_HEADERS = {"Albums", "Singles"}
    
    def __init__(self, artist, page):
        self.__artist = ArtistMetaInfo.__sanitize(artist)
        self.__page = page
        
        try:
            self.__channel_id, self.__playlist_ids = self.__read_cache()
        except FileNotFoundError:
            self.__channel_id, self.__playlist_ids = None, None
    
    def __get_cache_filename(self):
        return f'{self.__artist}.info'

    def __write_cache(self):
        pathlib.Path(CACHE_ROOT).mkdir(exist_ok=True)
        with open(f'{CACHE_ROOT}/{self.__get_cache_filename()}', 'wb') as f:
            pickle.dump(self.__channel_id, f)
            pickle.dump(self.__playlist_ids, f)

    def __read_cache(self):
        pathlib.Path(CACHE_ROOT).mkdir(exist_ok=True)
        with open(f'{CACHE_ROOT}/{self.__get_cache_filename()}', 'rb') as f:
            return pickle.load(f), pickle.load(f)

    @staticmethod
    def __sanitize(string):
        """
        Replace all substrings of non-alphanumeric characters with underscore
        """
        return re.sub('[^0-9a-zA-Z]+', '_', string)

    @staticmethod
    def __get_album_page_elements(driver):
        """
        Assumes that the driver is currently on some artist page
        Returns a list of elements corresponding to albums/singles
        """
        carousels = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "ytmusic-carousel-shelf-renderer")))
        carousels = [carousel for carousel in carousels if carousel.find_element_by_class_name("header").text in ArtistMetaInfo.__ALBUM_HEADERS]
        return [item for carousel in carousels for item in carousel.find_elements_by_tag_name("ytmusic-two-row-item-renderer")]

    @staticmethod
    def __get_album_playlist_ids(driver):
        """
        Assumes that the driver is currently on some artist page
        Returns a dictionary mapping album/single titles to their playlist ids
          for the current artist
        """
        playlist_ids = {}
    
        items = ArtistMetaInfo.__get_album_page_elements(driver)
        for i in range(len(items)):
            if i > 0:
                items = ArtistMetaInfo.__get_album_page_elements(driver)
        
            item = items[i]
            item_title = ArtistMetaInfo.__sanitize(item.find_element_by_class_name("details").find_element_by_class_name("title").text)
            item.click()

            WebDriverWait(driver, 10).until(EC.url_contains("playlist"))
            playlist_ids[item_title] = driver.current_url[driver.current_url.index('=') + 1:]
            driver.back()
        
        return playlist_ids

    def get_channel_info(self):
        """
        Returns the channel ID of the artist and a
            dictionary containing playlist name and playlist ID pairs
        """
        if self.__channel_id and self.__playlist_ids:
            return self.__channel_id, self.__playlist_ids
    
        try:
            driver = webdriver.Chrome()
            driver.get(self.__page)
        
            WebDriverWait(driver, 10).until(EC.url_contains("channel"))
            self.__channel_id = driver.current_url.split('/')[-1]
            self.__playlist_ids = ArtistMetaInfo.__get_album_playlist_ids(driver)
            
            self.__write_cache()
            return self.__channel_id, self.__playlist_ids
        finally:
            driver.quit()
        
        return "", {}

    def get_artist(self):
        return self.__artist
        

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
        }],
        'sleep_interval': 2,
        'max_sleep_interval': 5
    }
    if output_dir:
        ydl_opts['outtmpl'] = f'{DATA_ROOT}/{output_dir}/%(title)s.%(ext)s'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={video_id}' for video_id in video_ids])

        
def download_playlist_as_wav(playlist_id, output_dir=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }],
        'sleep_interval': 2,
        'max_sleep_interval': 5
    }
    if output_dir:
        ydl_opts['outtmpl'] = f'{DATA_ROOT}/{output_dir}/%(title)s.%(ext)s'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/playlist?list={playlist_id}'])


def main():
    queue = ArtistQueue("IU")
    for _ in range(2):
        artist, page = queue.get_next()
        artist_info = ArtistMetaInfo(artist, page)
        channel_id, playlist_ids = artist_info.get_channel_info()
        for playlist_name, playlist_id in playlist_ids.items():
            try:
                download_playlist_as_wav(playlist_id, output_dir=f"{artist_info.get_artist()}/{playlist_name}-{playlist_id}")
            except Exception as e:
                continue


if __name__ == '__main__':
    main()
