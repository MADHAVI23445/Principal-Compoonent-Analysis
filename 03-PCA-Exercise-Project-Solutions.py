#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright by Pierian Data Inc.</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Principal Component Analysis - Project

# -----
# -----
# 
# ## GOAL: Figure out which handwritten digits are most differentiated with PCA.
# 
# **Imagine you are working on an image recognition service for a postal service. It would be very useful to be able to read in the digits automatically, even if they are handwritten. (Quick note, this is very much how modern postal services work for a long time now and its actually more accurate than a human). The manager of the postal service wants to know which handwritten numbers are the hardest to tell apart, so he can focus on getting more labeled examples of that data. You will have a dataset of hand written digits (a very famous data set) and you will perform PCA to get better insight into which numbers are easily separable from the rest.**
# 
# -----
# -----

# # Data
# 
#     Background:
# 
#     E. Alpaydin, Fevzi. Alimoglu
#     Department of Computer Engineering
#     Bogazici University, 80815 Istanbul Turkey
#     alpaydin '@' boun.edu.tr
# 
# 
# #### Data Set Information from Original Authors:
# 
# We create a digit database by collecting 250 samples from 44 writers. The samples written by 30 writers are used for training, cross-validation and writer dependent testing, and the digits written by the other 14 are used for writer independent testing. This database is also available in the UNIPEN format.
# 
# We use a WACOM PL-100V pressure sensitive tablet with an integrated LCD display and a cordless stylus. The input and display areas are located in the same place. Attached to the serial port of an Intel 486 based PC, it allows us to collect handwriting samples. The tablet sends $x$ and $y$ tablet coordinates and pressure level values of the pen at fixed time intervals (sampling rate) of 100 miliseconds.
# 
# These writers are asked to write 250 digits in random order inside boxes of 500 by 500 tablet pixel resolution. Subject are monitored only during the first entry screens. Each screen contains five boxes with the digits to be written displayed above. Subjects are told to write only inside these boxes. If they make a mistake or are unhappy with their writing, they are instructed to clear the content of a box by using an on-screen button. The first ten digits are ignored because most writers are not familiar with this type of input devices, but subjects are not aware of this.
# 
# SOURCE: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

# ## Complete the Tasks in bold below
# 
# **TASK: Run the cells below to import the libraries and relevant data set.**

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


digits = pd.read_csv('../DATA/digits.csv')


# In[37]:


digits


# **TASK: Create a new DataFrame called *pixels* that consists only of the pixel feature values by dropping the number_label column.**

# In[ ]:


#CODE HERE


# In[38]:


pixels = digits.drop('number_label',axis=1)


# In[39]:


pixels


# ### Displaying an Image
# 
# **TASK: Grab a single image row representation by getting the first row of the pixels DataFrame.**

# In[ ]:


#CODE HERE


# In[40]:


single_image = pixels.iloc[0]


# In[41]:


single_image


# **TASK: Convert this single row Series into a numpy array.**

# In[ ]:


#CODE HERE


# In[42]:


single_image.to_numpy()


# **TASK: Reshape this numpy array into an (8,8) array.**

# In[ ]:


#CODE HERE


# In[43]:


single_image.to_numpy().shape


# In[44]:


single_image.to_numpy().reshape(8,8)


# **TASK: Use Matplotlib or Seaborn to display the array as an image representation of the number drawn. Remember your palette or cmap choice would change the colors, but not the actual pixel values.**

# In[45]:


#CODE HERE


# In[46]:


plt.imshow(single_image.to_numpy().reshape(8,8))


# In[47]:


plt.imshow(single_image.to_numpy().reshape(8,8),cmap='gray')


# In[48]:


sns.heatmap(single_image.to_numpy().reshape(8,8),annot=True,cmap='gray')


# ------
# 
# Now let's move on to PCA.

# ## Scaling Data

# **TASK: Use Scikit-Learn to scale the pixel feature dataframe.**

# In[49]:


#CODE HERE


# In[50]:


from sklearn.preprocessing import StandardScaler


# In[51]:


scaler = StandardScaler()


# In[52]:


scaled_pixels = scaler.fit_transform(pixels)


# In[53]:


scaled_pixels


# ## PCA
# 
# **TASK: Perform PCA on the scaled pixel data set with 2 components.**

# In[54]:


from sklearn.decomposition import PCA


# In[55]:


pca_model = PCA(n_components=2)


# In[57]:


pca_pixels = pca_model.fit_transform(scaled_pixels)


# **TASK: How much variance is explained by 2 principal components.**

# In[58]:


#CODE HERE


# In[59]:


np.sum(pca_model.explained_variance_ratio_)


# **TASK: Create a scatterplot of the digits in the 2 dimensional PCA space, color/label based on the original number_label column in the original dataset.**

# In[60]:


#CODE HERE


# In[61]:


plt.figure(figsize=(10,6),dpi=150)
labels = digits['number_label'].values
sns.scatterplot(pca_pixels[:,0],pca_pixels[:,1],hue=labels,palette='Set1')
plt.legend(loc=(1.05,0))


# **TASK: Which numbers are the most "distinct"?**

# In[62]:


# You should see label #4 as being the most separated group, 
# implying its the most distinct, similar situation for #2, #6 and #9.


# -----------
# ---------
# 
# ## Bonus Challenge 
# 
# **TASK: Create an "interactive" 3D plot of the result of PCA with 3 principal components. Lot's of ways to do this, including different libraries like plotly or bokeh, but you can actually do this just with Matplotlib and Jupyter Notebook. Search Google and StackOverflow if you get stuck, lots of solutions are posted online.**

# In[63]:


#CODE HERE


# In[64]:


from sklearn.decomposition import PCA


# In[65]:


pca_model = PCA(n_components=3)


# In[66]:


pca_pixels = pca_model.fit_transform(scaled_pixels)


# In[85]:


from mpl_toolkits import mplot3d


# In[90]:


plt.figure(figsize=(8,8),dpi=150)
ax = plt.axes(projection='3d')
ax.scatter3D(pca_pixels[:,0],pca_pixels[:,1],pca_pixels[:,2],c=df['number_label']);


# In[95]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[96]:


ax = plt.axes(projection='3d')
ax.scatter3D(pca_pixels[:,0],pca_pixels[:,1],pca_pixels[:,2],c=df['number_label']);


# 
