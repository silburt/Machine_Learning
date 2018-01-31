"""
This code builds and trains an LSTM model, with added flexibility so that I 
can conduct a grid search over hyperparameters. 
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding, TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

# one-hot encode on the fly, saves tons of memory
def one_hot_gen(X, Y, vocab_size, seq_length, batch_size=64):
    while True:
        for i in range(0, len(X), batch_size):
            x, y = X[i:i+batch_size].copy(), Y[i:i+batch_size].copy()
            x = np.eye(vocab_size)[x]
            y = np.eye(vocab_size)[y]
            yield (x, y)

# main routine
def train_model(genre, dir_model, MP):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used
    
    batch_size = MP['bs']
    lstm_size = MP['lstm_size']
    seq_length = MP['seq_length']
    drop = MP['dropout']
    epochs = MP['epochs']
    
    text_to_int, int_to_text, n_chars = np.load('playlists/%s/ancillary_char.npy'%genre)
    vocab_size = len(text_to_int)
    X = np.load('playlists/%s/X_sl%d_char.npy'%(genre, seq_length))
    y = np.load('playlists/%s/y_sl%d_char.npy'%(genre, seq_length))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = load_model(dir_model)
        print("successfully loaded previous model, continuing to train")
    except:
        print("generating new model")
        model = Sequential()
        model.add(GRU(lstm_size, dropout=drop, recurrent_dropout=drop, return_sequences=True,
                      input_shape=(seq_length, vocab_size)))
        for i in range(MP['n_layers']-1):
            model.add(GRU(lstm_size, dropout=drop, recurrent_dropout=drop, return_sequences=True))
        model.add(TimeDistributed(Dense(vocab_size, activation='softmax'))) #output shape=(bs, sl, vocab)

        decay = lr/epochs
        #optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay, clipvalue=1)
        optimizer = RMSprop(lr=MP['lr'], decay=decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print(model.summary())
    checkpoint = ModelCheckpoint(dir_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(one_hot_gen(X_train, y_train, vocab_size, seq_length, batch_size),
                        steps_per_epoch=len(X_train)/batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=one_hot_gen(X_test, y_test, vocab_size, seq_length, batch_size),
                        validation_steps=len(X_test)/batch_size)
    model.save(dir_model)

if __name__ == '__main__':
    genre = 'pop-rock-edm'
    
    # model parameters
    MP = {}
    MP['seq_length'] = 150              # sequence length
    MP['n_layers'] = int(sys.argv[1])   # number of lstm layers
    MP['lstm_size'] = int(sys.argv[2])  # lstm size
    MP['bs'] = int(sys.argv[3])         # batch size
    MP['dropout'] = float(sys.argv[4])  # dropout fraction
    MP['lr'] = 1e-3                     # learning rate
    MP['epochs'] = 50                   # n_epochs
    
    dir_model = 'models/%s_sl150_nl%d_size%d_bs%d_drop%.1f.h5'%(genre, MP['n_layers'],
                                                                MP['lstm_size'], MP['bs'],
                                                                MP['dropout'])
    
    train_model(genre, dir_model, MP)



