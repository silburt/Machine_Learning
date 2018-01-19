import numpy as np
import sys, glob
from utils.process_lyrics import *
from keras.models import load_model
import os

# From https://groups.google.com/forum/#!msg/keras-users/Y_FG_YEkjXs/nSLTa2JK2VoJ
# Francois Chollet: "It turns out that the 'temperature' for sampling (or more
# generally the choice of the sampling strategy) is critical to get sensible results.
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# text generation
def gen(genre, seq_len, temp, song):
    
    dir_lyrics = 'playlists/%s/'%genre
    dir_model = 'models/%s_sl%d_char.h5'%(genre, seq_len)
    
    model = load_model(dir_model)
    text_to_int, int_to_text, len_set = np.load('%sancillary_char.npy'%dir_lyrics)
    vocab_size = len(text_to_int)

    # open file and write pred
    name = song.split('/')[-1].split('.txt')[0]
    f = open('playlists/%s/pred/%s_sl%s_temp%.2f.txt'%(genre, name, seq_len, temp), 'w')

    # generate text
    lyrics = process_song(song)
    n_chars = len(lyrics)
    f.write(lyrics[:seq_len])
    pattern = [text_to_int[c] for c in list(lyrics[:seq_len])]
    print(''.join([int_to_text[c] for c in pattern]))
    print("****predicted lyrics for sl=%d, temp=%f:****"%(seq_len, temp))
    i, result = 0, ''
    while True:
        x = np.eye(vocab_size)[pattern].reshape(1,seq_len, vocab_size)
        preds = model.predict(x, verbose=0)
        pred = preds.reshape(seq_len, vocab_size)[-1]
        
        # sample
        index = sample(pred, temp)
        result = int_to_text[index]
        f.write(result)
        
        # update pattern
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        # break sequence
        if (i >= n_chars) and (result == '\n'):
            break
        i += 1
    print("\nDone.")
    f.close()

if __name__ == '__main__':
    n_songs = 1
    genre = 'pop-rock-edm'
    seq_length = 150
    temperatures = [0.6]
    #temperatures = [0.2,0.4,0.6,0.8,1.0,1.2]

    songs = glob.glob('playlists/%s/*.txt'%genre)
    for i in range(n_songs):
        for temp in temperatures:
            gen(genre, seq_length, temp, songs[i])
