from keras.datasets import mnist
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# training data
print(train_images.shape, "\n", len(train_labels), "\n", train_labels, "\n")
# test data
print(test_images.shape, "\n", len(test_labels), "\n", test_labels)

# build the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# compile the network
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# preparing the image data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# preparing the labels
train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)

# train the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# check how the model performs on the test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


