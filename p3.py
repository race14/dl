

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np





cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()





x_train.shape





x_test.shape





x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)





classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]





def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])




plot_sample(x_train, y_train, 0)





model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)





test_acc, test_loss = model.evaluate(x_test, y_test)
print('accuracy is', test_loss)
print('loss is ',test_acc)




y_pred = model.predict(x_test)





y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]



y_test[:5]



plot_sample(x_test, y_test,3)


classes[y_classes[3]]







