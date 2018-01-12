#https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

from collections import Counter
import numpy as np
import glob
import sys
from utils.process_lyrics import *
from keras.utils import np_utils

def get_embedding_index(embed_dim=50):
    f = open('utils/glove.6B/glove.6B.%dd.txt'%embed_dim,'r',encoding='utf-8')
    
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
def get_embedding_matrix(embeddings_index,text_to_int,embed_dim):
    # all text_to_int values by this point should be in embeddings_index
    
    # embed matrix has +1 dimension for the zero vector. I.e. end of sentences
    # are padded with the zero embed vector. 
    embedding_matrix = np.zeros((len(text_to_int)+1, embed_dim))
    for word, i in text_to_int.items():
        embedding_matrix[i] = embeddings_index.get(word)

    return embedding_matrix

def main(genre,n_songs,seq_length,word_or_character,min_word_occurrence=2,embed_dim=50):
    
    # get song lyrics
    dir_lyrics = 'playlists/%s/'%genre
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs, song_names, n_songs = [], [], len(files)
    for i,f in enumerate(files):
        songs.append(process_song(f, word_or_character))
        song_names.append(f)

    # prepare word/character corpus
    if word_or_character == 'character':
        set_ = sorted(list(set(' '.join(songs))))
    elif word_or_character == 'word':
        set_ = []
        embeddings_index = get_embedding_index()
        for s in songs:
            set_ += s
        set_ = Counter(set_)                    #gets unique sorted dictionary
        for k in list(set_):
            # delete rare/non-gloVe words from corpus
            if (set_[k] < min_word_occurrence) or (embeddings_index.get(k) == None):
                del set_[k]
        set_, vals = zip(*set_.most_common())

    # get char/word to int mappings and vice versa.
    len_set = len(set_)      #number of unique words/chars
    text_to_int = dict((c, i) for i, c in enumerate(set_))
    int_to_text = dict((i, c) for i, c in enumerate(set_))
    np.save('%sancillary_%s.npy'%(dir_lyrics,word_or_character),[text_to_int,int_to_text,len_set])
    if word_or_character == 'word':
        embedding_matrix = get_embedding_matrix(embeddings_index,text_to_int,embed_dim)
        np.save('%sembedding_matrix_%dd.npy'%(dir_lyrics,embed_dim),embedding_matrix)

    print(text_to_int)
    print(int_to_text)

    # get data arrays for training LSTMs
    # !!! split into sentences and pad short sentences with "zeros", but not 0 cause that maps to an embed vector.
    for sl in seq_length:
        dataX, dataY, data_songnames = [], [], []
        for i in range(n_songs):
            lyric = songs[i]
            for j in range(0,len(lyric)-sl):
                seq_in = lyric[j:j + sl]
                seq_out = lyric[j + sl]
                try:
                    t2i_i = [text_to_int[text] for text in seq_in]
                    t2i_o = text_to_int[seq_out]
                    dataX.append(t2i_i)
                    dataY.append(t2i_o)
                    data_songnames.append(song_names[i])
                except:
                    # a sparse word->int set (rare words removed) is
                    # going to yield words with no matches
                    pass
        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)

        # make X and y datasets
        X = np.asarray(dataX)
        if word_or_character == 'character':
            X = np.reshape(dataX, (n_patterns,sl,1))    # reshape X:[samples,time steps,features]
            X = X / float(len_set)                      # normalize
            y = np_utils.to_categorical(dataY)          # 1-hot encode the output variable
        elif word_or_character == 'word':
            y = np.zeros((len(dataY), embed_dim))
            np.save('%syraw_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),np.asarray(dataY))
            for i in range(len(dataY)):
                y[i] = embedding_matrix[dataY[i]]

        # save data
        np.save('%sX_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),X)
        np.save('%sy_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),y)
        np.save('%ssong_names_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),data_songnames)

if __name__ == '__main__':
    n_songs = -1
    seq_length = [25,50,75,100,125,150,175,200]
    #seq_length = [4]#,6,8,10,12,15]
    word_or_character = 'character'

    #genre = 'country'
    genre = 'pop-rock-edm'

    main(genre,n_songs,seq_length,word_or_character)
