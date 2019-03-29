from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn import preprocessing
import numpy

seed = 7
numpy.random.seed(seed)

# Load dataset
dataset = numpy.genfromtxt("balance-scale.csv", delimiter=",", dtype='str')
X = dataset[:][:,1:5].astype(numpy.float)
Y = dataset[:][:,0]

encoder = preprocessing.LabelBinarizer()
Y = encoder.fit_transform(Y)
# Alternatively:
# Y = keras.utils.to_categorical(Y, num_classes=3, dtype='str')
# print(Y)
# print(X)

# Define and Compile
model = Sequential()
model.add(Dense(18, input_dim=4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=300, batch_size=6, verbose=1)

# Evaluate the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))