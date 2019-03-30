from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import preprocessing
from keras import optimizers
import numpy
import keras

seed = 7
numpy.random.seed(seed)
max_words = 50

# define documents (list of tweets)
dataset = numpy.genfromtxt("Datasets\\sentiment140_lessreduced.txt", delimiter="\t", dtype='str')
X = dataset[:,5]
y = dataset[:,0].astype(numpy.float)

# print(X.shape)
# print(Y.shape)
# print(Y)

# Create tokenizer
t = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

# Fit the tokenizer on the data
t.fit_on_texts(X)

# Summarize fit data
# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)

# print(encoded_X)

# Preprocess input data
X_train = X[0:80000]
X_test = X[80000:100000]
# print(X_train.shape)
# print(X_test.shape)

# Integer encode documents
X_train = t.texts_to_matrix(X_train, mode='count')
X_test = t.texts_to_matrix(X_test, mode='count')


# Preprocess labels
encoder = preprocessing.LabelBinarizer()
Y = encoder.fit_transform(y)
Y = np_utils.to_categorical(Y, num_classes=2, dtype='float32')

Y_train = Y[0:80000,:]
Y_test = Y[80000:100000,:]
# print(Y_train.shape)
# print(Y_test.shape)
# print(Y)


# Define and Compile
e_init = keras.initializers.RandomUniform(-0.01, 0.01, seed=1)
init = keras.initializers.glorot_uniform(seed=1)
embed_vec_len = 32  # values per word

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, embeddings_initializer=e_init, mask_zero=True))
model.add(LSTM(units=100, kernel_initializer=init, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer=init, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, Y_train, epochs=10, batch_size=150, verbose=1)

# Evaluate the model
scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save(".\\Models\\Tweet_Model.h5")