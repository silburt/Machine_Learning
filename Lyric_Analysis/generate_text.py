import numpy as np
import sys, glob
from utils.process_lyrics import *
from keras.models import load_model

# text generation
def gen(genre,dir_model,seq_length,word_or_character,embed_dim=50):
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    text_to_int, int_to_text, len_set = np.load('%sancillary_%s.npy'%(dir_lyrics,word_or_character))
    
    #generate initial seed
    songs = glob.glob('%s/*.txt'%dir_lyrics)
    pattern = None
    while pattern == None:
        seed = np.random.randint(0, len(songs))
        ini = list(process_song(songs[seed],word_or_character)[:seq_length])
        try:
            pattern = [text_to_int[c] for c in list(ini)]
        except:
            # sometimes we choose a lyric with rare words
            # not in the text_to_int conversion.
            pass
    print(songs[seed])
    
    # generate text
    if word_or_character == 'character':
        print(''.join([int_to_text[c] for c in pattern]))
        print("****predicted lyrics:****")
        for i in range(300):
            x = np.reshape(pattern, (1, seq_length, 1))
            x = x / float(len_set)
            pred = model.predict(x, verbose=0)
            #index = np.argmax(pred)
            index = np.random.choice(len(pred[0]), p=pred[0])
            result = int_to_text[index]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    elif word_or_character == 'word':
        print(' '.join([int_to_text[c] for c in pattern]))
        print("****predicted lyrics:****")
        em = np.load('%sembedding_matrix_%dd.npy'%(dir_lyrics,embed_dim))
        em_norms = np.sqrt(np.sum(em*em,axis=1))
        em_norms[em_norms==0] = -1
        matrix_abs = np.abs(em)
        labels = list(text_to_int.keys())
        for i in range(50):
            x = np.reshape(pattern, (1, seq_length))
            pred = model.predict(x, verbose=0)
            pred_norm = np.sqrt(np.sum(pred*pred))
            proj = np.sum(pred*em,axis=1)/(em_norms*pred_norm) #cosine similarity
            #index = np.argmax(proj)
            
            #probs = proj - np.min(proj)
            #probs /= np.sum(probs)
            #index = np.random.choice(em.shape[0], p=probs)
            proj[proj<0] = 0
            probs = proj/np.sum(proj)
            index = np.random.choice(em.shape[0], p=probs)

            result = labels[index]
            sys.stdout.write(' %s'%result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    print("\nDone.")

if __name__ == '__main__':
    genre = 'rock'
    word_or_character = 'word'
    seq_lengths = [4,6,8,10,15]

    for seq_length in seq_lengths:
        print('***********seq_length=%d***********'%seq_length)
        dir_model = 'models/%s_sl%d_%s.h5'%(genre,seq_length,word_or_character)
        gen(genre,dir_model,seq_length,word_or_character)
