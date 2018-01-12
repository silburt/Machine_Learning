#bulk of code from
#https://bigishdata.com/2016/09/27/getting-song-lyrics-from-geniuss-api-scraping/

#Helpful links
#http://www.jw.pe/blog/post/quantifying-sufjan-stevens-with-the-genius-api-and-nltk/
#https://stackoverflow.com/questions/45647769/genius-api-search-result-successful-but-no-hits
#https://stackoverflow.com/questions/9942594/unicodeencodeerror-ascii-codec-cant-encode-character-u-xa0-in-position-20/28152554

#importing spotify playlists
#https://github.com/watsonbox/exportify

import requests
from bs4 import BeautifulSoup
import pandas as pd
import glob
import re

base_url = 'http://api.genius.com'
auth = open('utils/Authorization.txt','r').readlines()
headers = {'Authorization': 'Bearer %s'%auth[2].split(':')[1]}

def clean_names(artist, song):
    artist = artist.split(',')[0]   #only take first of multiple artists
    song = song.split('(')[0]       #exclude (feat. ) or other details
    song = song.split('-')[0]       #exclude extra details
    song = song.split('/')[0]       #exclude extra details
    song = song.rstrip()
    ch = ['?','!']
    for c in ch:
        song = song.replace(c,'')
    return artist.encode("utf-8"), song.encode("utf-8")

def extract_lyrics(song_api_path):
    song_url = base_url + song_api_path
    response = requests.get(song_url, headers=headers)
    json = response.json()
    path = json["response"]["song"]["path"]
    page_url = "http://genius.com" + path
    page = requests.get(page_url)
    html = BeautifulSoup(page.text, "html.parser")
    lyrics = html.find("div", class_="lyrics").get_text()
    return lyrics.encode("utf-8")

def get_song_lyrics(artist, song):
    try:
        artist, song = clean_names(artist, song)
    except:
        "bad entry, couldnt clean song entry"
        return None, artist, song
    search_url = base_url + "/search"
    
    #try song then artist
    response = requests.get(search_url, params={'q': song}, headers=headers)
    json = response.json()
    for hit in json["response"]["hits"]:
        if artist in hit["result"]["primary_artist"]["name"].encode("utf-8"):
            return extract_lyrics(hit["result"]["api_path"]), artist, song

    #try artist then song
    response = requests.get(search_url, params={'q': artist}, headers=headers)
    json = response.json()
    for hit in json["response"]["hits"]:
        if song in hit["result"]["title"].encode("utf-8"):
            return extract_lyrics(hit["result"]["api_path"]), artist, song

    #last desperate attempt, except for edm this works most often...
    try:
        page_url = "http://genius.com/%s-%s-lyrics"%(artist.replace(' ','-'),song.replace(' ','-'))
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, "html.parser")
        lyrics = html.find("div", class_="lyrics").get_text()
        return lyrics.encode("utf-8"), artist, song
    except:
        return None, artist, song

###### Main Loop ######
if __name__ == '__main__':

    dir = 'playlists/rap/'
    tracks = pd.read_csv(glob.glob('%s*.csv'%dir)[0], encoding = "ISO-8859-1")
    
    # Main Loop
    skipped_tracks = 0
    for index, track in tracks.iterrows():
        print(index)
        lyrics, artist, song = get_song_lyrics(track['Artist Name'], track['Track Name'])
        if lyrics:
            #write to txt file
            artist = artist.decode("utf-8").replace(' ','_').replace('/','')
            song = song.decode("utf-8").replace(' ','_')
            with open('%s%s-%s.txt'%(dir,artist,song), 'wb') as out:
                out.write(lyrics)
        else:
            print("Couldnt find: %s - %s"%(artist, song))
            skipped_tracks += 1

    n_tracks = len(tracks)
    print("successfully processed %d of %d tracks"%(n_tracks-skipped_tracks, n_tracks))
