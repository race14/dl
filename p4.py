#!/usr/bin/env python
# coding: utf-8
# Use Autoencoder to implement anomaly detection. Build the model by using: 
a. Import required libraries 
b. Upload / access the dataset 
c. Encoder converts it into latent representation 
d. Decoder networks convert it back to the original input 
e. Compile the models with Optimizer, Loss, and Evaluation Metrics
# In[1]:


import keras
from keras import layers

encoding_dim = 32  
input_img = keras.Input(shape=(784,))

encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = keras.Model(input_img, decoded)


# In[2]:


encoder = keras.Model(input_img, encoded)


# In[4]:


encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


# In[5]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[6]:


from keras.datasets import mnist

import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


# In[7]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# In[8]:


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[8]:


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[7]:


# Use Matplotlib

import matplotlib.pyplot as plt
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

