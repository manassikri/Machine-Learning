#!/usr/bin/env python
# coding: utf-8

# In[5]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[6]:


fashion_mnist = keras.datasets.fashion_mnist


# In[7]:


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[8]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[9]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[10]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[11]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[12]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[13]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


model.fit(train_images, train_labels, epochs=10)


# In[21]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[16]:


predictions = model.predict(test_images)


# In[17]:


np.argmax(predictions[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




