#!/usr/bin/env python
# coding: utf-8

# ### Small Image Classification Using Convolutional Neural Network (CNN)

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()


# In[3]:


x_train.shape


# In[4]:


x_test.shape


# In[10]:


x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)


# In[11]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[12]:


def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


# <h4 style="color:purple">Now let us build a convolutional neural network to train our images</h4>

# In[13]:


plot_sample(x_train, y_train, 0)


# In[12]:


model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[13]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[14]:


model.fit(x_train, y_train, epochs=5)


# In[22]:


test_acc, test_loss = model.evaluate(x_test, y_test)
print('accuracy is', test_loss)
print('loss is ',test_acc)


# In[16]:


y_pred = model.predict(x_test)


# In[17]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[18]:


y_test[:5]


# In[19]:


plot_sample(x_test, y_test,3)


# In[20]:


classes[y_classes[3]]


# In[ ]:




