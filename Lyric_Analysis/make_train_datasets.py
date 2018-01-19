import numpy as np
import glob
import sys
from utils.process_lyrics import *
from keras.utils import np_utils

def main(genre, n_songs, seq_length):
    
    # get song lyrics
    dir_lyrics = 'playlists/%s/'%genre
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs, song_names, n_songs = [], [], len(files)
    for i,f in enumerate(files):
        song = process_song(f)
        #print(song)
        songs.append(song)
        song_names.append(f)

    # prepare word/character corpus
    set_ = sorted(list(set(' '.join(songs))))

    # get char/word to int mappings and vice versa.
    len_set = len(set_)      #number of unique words/chars
    text_to_int = dict((c, i) for i, c in enumerate(set_))
    int_to_text = dict((i, c) for i, c in enumerate(set_))
    np.save('%sancillary_char.npy'%dir_lyrics,[text_to_int,int_to_text,len_set])

    print(text_to_int)
    print(int_to_text)

    for sl in seq_length:
        # get data arrays for training TimeDistributed LSTMs
        dataX, dataY, data_songnames = [], [], []
        for i in range(n_songs):
            lyric = songs[i]
            for j in range(0,len(lyric)-sl-1):
                seq_in = lyric[j:j+sl]
                seq_out = lyric[j+1:j+sl+1]
                try:
                    t2i_i = [text_to_int[text] for text in seq_in]
                    t2i_o = [text_to_int[text] for text in seq_out]
                    dataX.append(t2i_i)
                    dataY.append(t2i_o)
                    data_songnames.append(song_names[i])
                except:
                    # a sparse word->int set (rare words removed) is
                    # going to yield words with no matches
                    pass
        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)

        # (1-hot encode via a generator in keras)
        X = np.asarray(dataX)
        y = np.asarray(dataY)

        # save data
        np.save('%sX_sl%d_char.npy'%(dir_lyrics,sl), X)
        np.save('%sy_sl%d_char.npy'%(dir_lyrics,sl), y)
        np.save('%ssong_names_sl%d_char.npy'%(dir_lyrics,sl), data_songnames)

if __name__ == '__main__':
    n_songs = -1
    seq_length = [25,50,75,100,125,150,175,200]
    #seq_length = [6]#,6,8,10,12,15]

    #genre = 'country'
    genre = 'pop-rock-edm'

    main(genre, n_songs, seq_length)
