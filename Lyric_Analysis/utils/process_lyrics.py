import unidecode
import re

def unicodetoascii(text, word_or_character):
    if word_or_character == 'character':
        TEXT = (text.
                replace('\xe2\x80\x99', "'").
                replace('\x92',"'").
                replace('\xe2\x80\x8be', 'e').
                replace('\xc3\xa9', 'e').
                replace('\xc2\x92',"'").
                replace('\xe2\x80\x90', '-').
                replace('\xe2\x80\x91', '-').
                replace('\xe2\x80\x92', '-').
                replace('\xe2\x80\x93', '-').
                replace('\xe2\x80\x94', '-').
                replace('\xe2\x80\x94', '-').
                replace('\xe2\x80\x98', "'").
                replace('\xe2\x80\x9b', "'").
                replace('\xe2\x80\x9c', '"').
                replace('\xe2\x80\x9c', '"').
                replace('\xe2\x80\x9d', '"').
                replace('\xe2\x80\x9e', '"').
                replace('\xe2\x80\x9f', '"').
                replace('\xe2\x80\xa6', '...').
                replace('\xe2\x80\xb2', "'").
                replace('\xe2\x80\xb3', "'").
                replace('\xe2\x80\xb4', "'").
                replace('\xe2\x80\xb5', "'").
                replace('\xe2\x80\xb6', "'").
                replace('\xe2\x80\xb7', "'").
                replace('\xe2\x81\xba', "+").
                replace('\xe2\x81\xbb', "-").
                replace('\xe2\x81\xbc', "=").
                replace('\xe2\x81\xbd', "(").
                replace('\xe2\x81\xbe', ")").
                replace('\xe2\x80\x8b', '').
                replace('\xc3\xa2\xe2\x82\xac\xcb\x9c',"'").
                replace('\xc3\xa4','a').
                replace('\xc3\xb1','n').
                replace('\xc3\xb3','o').
                replace('_',' ').
                replace('*',' ').
                replace('+','and').
                replace('{','(').
                replace('}',')').
                replace('[','(').
                replace(']',')').
                replace('`',"'").
                replace('"',"'").
                replace('$','').
                replace('&','and').
                replace('#','number ').
                replace('%', 'percent').
                replace('\n\n','\n').
                replace('/',' and ')
                )
    
    elif word_or_character == 'word':
        TEXT = (text.replace(',','').
                replace('?',' . ').
                replace('!',' . ').
                replace('.',' . ').
                replace(',','').
                replace(';','').
                replace('…',' ').
                #replace('\n',' \n ').
                replace('-',' ').
                replace('"','').
                replace('`','').
                replace("'",'')
                )
    return TEXT

def process_song(song_dir, word_or_character='character'):
    song = open(song_dir,'r',encoding='utf-8').read().lower()
    song = unidecode.unidecode(unicodetoascii(song, word_or_character))
    song = re.sub("[\(\[].*?[\)\]]", "", song)
    if word_or_character == 'word':
        return song.split()
        #return re.split("(\n)",song)
    return song
