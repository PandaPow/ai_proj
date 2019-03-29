from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers
from sklearn import preprocessing
from matplotlib import pyplot
import numpy

seed = 7
numpy.random.seed(seed)

# Load dataset (60,000 training, 10,000 test)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape)
# pyplot.imshow(X_train[0])
# pyplot.show()

# Reshape input data (1x RGB channel)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess labels
Y_train = np_utils.to_categorical(y_train, num_classes=10, dtype='float32')
Y_test = np_utils.to_categorical(y_test, num_classes=10, dtype='float32')

# Define and Compile
model = Sequential()

model.add(Conv2D(32, [3, 3], activation='relu', input_shape=(1,28,28), data_format="channels_first"))
model.add(Conv2D(32, [3, 3], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('Image_Test_Model.h5')