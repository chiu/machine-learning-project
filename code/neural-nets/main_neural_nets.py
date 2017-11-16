
#starting with neural nets.

import pandas as pd


train = pd.read_csv("../../data/train.csv")

print(train)




import keras
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential



model = Sequential()

# hidden layer 1
# hidden layer with 64 nodes
model.add(Dense(units=32, input_dim=x_train.shape[1], activation='relu'))

# reduce over fitting, causes any given node to fail 20% of the time so that it is not overly relied
# on by the neural net
model.add(Dropout(rate=0.5))

# output layer
model.add(Dense(units=1, activation='relu'))
# model.add(Activation('softmax'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())

print(x_train[:2])
print(y_train[:2])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(loss_and_metrics)
print('finished')
