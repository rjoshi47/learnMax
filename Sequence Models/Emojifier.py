import numpy as np
np.random.seed(0)
from emo_utils import *
import emoji
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding 
    to words in the sentences. The output shape should be such that it can be given 
    to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0] # number of training examples
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):  # loop over training examples
        sentence_words =X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j +=1
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    '''
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    '''
    
    vocab_len = len(word_to_index) + 1 # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0] # define dimensionality of your GloVe word vectors (= 50)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    '''
    Set each row "index" of the embedding matrix to be the word vector 
    representation of the "index"th word of the vocabulary
    '''
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    '''
    Define Keras embedding layer with the correct output/input sizes, make it trainable. 
    Use Embedding(...). Make sure to set trainable=False. 
    '''
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    
    '''
    Build the embedding layer, it is required before setting the weights of the embedding layer. 
    Set the weights of the embedding layer to the embedding matrix. 
    '''
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into 
            its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices 
            in the vocabulary (400,001 words)

    Returns: model -- a model instance in Keras
    """
    
    '''
    Define sentence_indices as the input of the graph, 
    it should be of shape input_shape and dtype 'int32' (as it contains indices).
    '''
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)    
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X =  Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X =  Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    return model

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=len).split())

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 10)                0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 10, 50)            20000050  
_________________________________________________________________
lstm_3 (LSTM)                (None, 10, 128)           91648     
_________________________________________________________________
dropout_3 (Dropout)          (None, 10, 128)           0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
_________________________________________________________________
activation_2 (Activation)    (None, 5)                 0         
=================================================================
'''

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("Test accuracy = ", acc)

x_test = np.array(['Lets play'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
# output: Lets play ⚾

'''
Some predictions by model
Input               Output
want to eat food    🍴
She loves me        ❤️
I got new job       😄
I lost my phone     😞
'''
