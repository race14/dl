
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical




cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()



x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_t`est = to_categorical(y_test)


base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)


for layer in base.layers:
 layer.trainable = False


model = Sequential()
model.add(base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax')) 


model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5 )


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

