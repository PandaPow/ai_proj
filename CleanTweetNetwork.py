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
import time
import datetime

ts = time.time()
startTime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M')

seed = 7
numpy.random.seed(seed)
max_words = 350
num_tags = 7

# define documents (list of tweets)
dataset = numpy.genfromtxt("Datasets\\sentiment140_cleanoutput.txt", delimiter="\t", dtype='str')
X = dataset[:,0:2]
y = dataset[:,2].astype(numpy.float)

# print(X.shape)
# print(Y.shape)
# print(Y)

# Create tokenizer
tweet_token = Tokenizer(num_words=max_words)
tag_token = Tokenizer(num_words=num_tags)

# Fit the tokenizer on the data
tweet_token.fit_on_texts(X[:,0])
tag_token.fit_on_texts(X[:,1])

# Summarize fit data
# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)

# print(encoded_X)

# Preprocess input data
data_size = X.shape[0]
train_size = int(0.8*data_size)
test_size = data_size - train_size
# X_train = X[0:80000]
# X_test = X[80000:100000]
# print(X_train.shape)
# print(X_test.shape)

X_train = numpy.zeros((train_size,(max_words+num_tags)),dtype='str')
X_test = numpy.zeros((test_size,(max_words+num_tags)),dtype='str')

# Integer encode documents
X_train[:,0:max_words] = tweet_token.texts_to_matrix(X[0:train_size,0], mode='count')
X_train[:,max_words:(max_words+num_tags)] = tag_token.texts_to_matrix(X[0:train_size,1], mode='count')
X_test[:,0:max_words] = tweet_token.texts_to_matrix(X[train_size:data_size,0], mode='count')
X_test[:,max_words:(max_words+num_tags)] = tag_token.texts_to_matrix(X[train_size:data_size,1], mode='count')


# Preprocess labels
encoder = preprocessing.LabelBinarizer()
Y = encoder.fit_transform(y)
Y = np_utils.to_categorical(Y, num_classes=2, dtype='float32')

Y_train = Y[0:train_size,:]
Y_test = Y[train_size:data_size,:]
# print(Y_train.shape)
# print(Y_test.shape)
# print(Y)

filename = ".\\Logs\\"+startTime+"_tweet_training_data.csv"

# Setup Callbacks
callbackList = [
	keras.callbacks.CSVLogger(filename,separator=',',append=True),
	keras.callbacks.ModelCheckpoint(".\\Models\\"+startTime+"_Tweet_Model_Best.h5", monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1),
	keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.002, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=False)
	]

# Define and Compile
e_init = keras.initializers.RandomUniform(-0.01, 0.01, seed=1)
init = keras.initializers.glorot_uniform(seed=1)
embed_vec_len = 32  # values per word

model = Sequential()
# model.add(Embedding(input_dim=max_words+num_tags, output_dim=32, embeddings_initializer=e_init, mask_zero=True))
# model.add(LSTM(units=100, kernel_initializer=init, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, input_dim=max_words+num_tags,kernel_initializer=init,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, kernel_initializer=init, activation='relu'))
# model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer=init, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, callbacks = callbackList)

# Evaluate the model
scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save(".\\Models\\"+startTime+"_Tweet_Model_Last.h5")

with open(filename,'a',newline='') as outputfile:
	outputwriter = csv.writer(outputfile,delimiter=',')
	outputwriter.writerow([''])
	outputwriter.writerow(['',scores[0],scores[1]])