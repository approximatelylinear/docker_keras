

########### Data ##############
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn import cross_validation

mnist = fetch_mldata('MNIST original', data_home='/opt/data')
print("Fetched MNIST data.")
labels = np.unique(mnist.target)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    mnist.data, mnist.target, test_size=0.4, random_state=0)
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
print(mnist.data.shape)
print(mnist.target.shape)
print(labels)
print('-'*50)


########## Keras ###############

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical

model = Sequential()
model.add(Dense(output_dim=64, input_dim=X_train.shape[1]))
model.add(Activation("relu"))
model.add(Dense(output_dim=labels.shape[0]))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train_binary, nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, y_test_binary, batch_size=32)
print("loss: {}\naccuracy: {}".format(loss_and_metrics[0], loss_and_metrics[1]))
