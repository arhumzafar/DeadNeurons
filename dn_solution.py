# One way we can solve the problem in the two files is by using LeakyReLU.
# Unlike the standard ReLU, the Leaky ReLU has a small slope 
# for negative values, as an attempt to fix the "dying ReLU" problem.
# One advantage here w/ the Leaky ReLU is that you will not have
# a vanishing gradient.

# What are "Dead Neurons"?
#   when we train a neural network improperly, as a result, some neurons die
#   and produce unchangable activation, and never revive.

#   The main reason for "dead neurons" is that neurons run into the situation 
#   that always produce specific values and have zero gradient.


# Once again, we can use Leaky ReLU to solve this problem :)

from keras.layers.advanced_activations import LeakyReLU
net = Dense(2, activation=LeakyReLU(), ...))(net_input)
