import DnnLib

#layer = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
#layer.SGD(learning_rate=0.01)

optimizer = DnnLib.Adam(learning_rate=0.01)
