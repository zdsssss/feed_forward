from nn import activations, loss_f
from nn import network
from nn import layers
from nn import xoptimzers
import numpy as np

# x_train = np.array([
#     [0,1],
#     [1,2],
#     [2,3],
#     [3,4],
#     [4,5]
# ])

# y_train = np.array([
#     [1],
#     [3],
#     [5],
#     [7],
#     [9]
# ])

x_train = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,6],
    [6,7]
])

y_train = np.array([
    [3],
    [5],
    [7],
    [9],
    [11],
    [13]
])

net = network.network()
net.add(layers.layer_dense(2,8, activation=activations.relu()))
net.add(layers.layer_dense(8,1, activation=activations.relu()))

net.fit(
    x_train=x_train, 
    y_train=y_train, 
    ep=50000, 
    optimizer=xoptimzers.gradient_descent(),
    loss_function=loss_f.Mse(),
)

# print()
# print("-"*100)
# for i in x_train:
#     print(f"{i} : ",net.predict_debug(np.array(i)))
#     print("-"*100)
# print()
for i in x_train:
    print(f"{i} {net.predict(i)}")
# a = net.debug()
# for i in a:
#     print(i.weights)