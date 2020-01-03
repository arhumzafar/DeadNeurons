# Now, let's build a classifier with one hidden layer.
# Additionally, let's initilize "b" with a very negative value, in order to simulate a scenario caused by a huge learning rate or improper weight initialization.

import keras
from keras.initilizers import Constant
from keras.layers import Input, Dense
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

b_init = -0.5  # [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
net_input = Input(shape=(4,))
net = Dense(2, activation='relu', bias_initializer=Constant(b_init))(net_input)
net = Dense(1, activation='sigmoid')(net)

model = Model(net_input, net)
model.compile(optimizer=SGD(0.5), loss='binary_crossentropy', metrics=['accuracy'])

"""
_b_init = K.eval(model.trainable_weights[1][0])
assert abs(_b_init - b_init) < 0.001
"""
result = model.fit(x, y, epochs=10)
acc = result.history['acc'][-1]