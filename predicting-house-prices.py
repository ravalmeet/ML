#!/usr/bin/env python
# coding: utf-8

# <hr/>
# 
# # Predicting House Prices (Keras - Artificial Neural Network)
# **[by Tomas Mantero](https://www.kaggle.com/tomasmantero)**
# <hr/>
# 
# ### Table of Contents
# 1. [Overview](#ch1)
# 1. [Dataset](#ch2)
# 1. [Exploratory Data Analysis](#ch3)
# 1. [Working with Feature Data](#ch4)
# 1. [Feature Engineering](#ch5)
# 1. [Scaling and Train Test Split](#ch6)
# 1. [Creating a Model](#ch7)
# 1. [Training the Model](#ch8)
# 1. [Evaluation on Test Data](#ch9)
# 1. [Predicting on a Brand New House](#ch10)

# <a id="ch1"></a>
# ## Overview 
# <hr/>
# 
# One of the objectives of this notebook is to **show step-by-step how to analyze and visualize the dataset to predict future home prices.** Moreover, we are going to explain most of the concepts used so that you understand why we are using them.
# In base of features like sqft_living, bathrooms, bedrooms, view, and others, we are going to build a deep learning model that can predict future price houses. 
# 
# If you have a question or feedback, do not hesitate to write and if you like this kernel,<b><font color='green'> please upvote! </font></b>
# 
# <img src="https://images.pexels.com/photos/1546168/pexels-photo-1546168.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260" title="source: www.pexels.com" />
# <br>
# 
# The following questions will be answered throughout the Kernel:
# * ***Which features are available in the dataset?***
# * ***Which features are categorical?***
# * ***Which features are numerical?***
# * ***Which features contain blank, null or empty values?***
# * ***What are the data types for various features?***
# * ***What is the distribution of numerical feature values across the samples?***
# * ***Which features are more correlated to the price?***

# <a id="ch2"></a>
# ## Dataset
# <hr/>
# 
# * This dataset contains house sale prices for King County, which includes Seattle. 
# * It includes homes sold between May 2014 and May 2015.
# * 21 columns. (features)
# * 21597 rows.
# 
# ***Feature Columns***
#     
# * **id:** Unique ID for each home sold
# * **date:** Date of the home sale
# * **price:** Price of each home sold
# * **bedrooms:** Number of bedrooms
# * **bathrooms:** Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * **sqft_living:** Square footage of the apartments interior living space
# * **sqft_lot:** Square footage of the land space
# * **floors:** Number of floors
# * **waterfront:** - A dummy variable for whether the apartment was overlooking the waterfront or not
# * **view:** An index from 0 to 4 of how good the view of the property was
# * **condition:** - An index from 1 to 5 on the condition of the apartment,
# * **grade:** An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * **sqft_above:** The square footage of the interior housing space that is above ground level
# * **sqft_basement:** The square footage of the interior housing space that is below ground level
# * **yr_built:** The year the house was initially built
# * **yr_renovated:** The year of the house’s last renovation
# * **zipcode:** What zipcode area the house is in
# * **lat:** Lattitude
# * **long:** Longitude
# * **sqft_living15:** The square footage of interior housing living space for the nearest 15 neighbors
# * **sqft_lot15:** The square footage of the land lots of the nearest 15 neighbors

# ### Imports

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

# evaluation on test data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('kc_house_data.csv')


# ### Acquire data
# The Python Pandas packages helps us work with our datasets. We start by acquiring the datasets into Pandas DataFrames.

# In[ ]:





# ### Analyze by describing data
# Pandas also helps describe the datasets answering following questions early in our project.
# 
# ***Which features are available in the dataset?***

# In[2]:


print(df.columns.values)


# ***Which features are categorical?***
# 
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# * Categorical: id, waterfront, zipcode.
# 
# ***Which features are numerical?***
# 
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# * Continous: price, bathrooms, floors, lat, long.
# * Discrete: date, bedrooms, sqft_living, sqft_lot, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, sqft_living15, sqft_lot15.

# In[3]:


# preview the data
df.head()


# In[4]:


# preview the data
df.tail()


# ***Which features contain blank, null or empty values?***
# 
# We can check for missing values with pandas isnull(). This indicates whether values are missing or not. Then we can sum all the values to check every column. 

# In[5]:


# No missing values
df.isnull().sum()


# ***What are the data types for various features?***
# 
# Five features are floats, fifteen are integers and one is an object.

# In[6]:


df.info()


# ***What is the distribution of numerical feature values across the samples?***

# In[7]:


df.describe().transpose()


# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# ### Correlating
# 
# We want to know how well does each feature correlate with Price. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# ### Completing
# 
# Since there are no missing values we do not need to complete any values. 
# 
# ### Correcting
# 
# Id feature may be dropped from our analysis since it does not add value.
# Date feature may be dropped since we are going to do feature engineering and make a year and month column.
# Zipcode feature is a special case, we could use it, but since we do not know exactly the zones of King County we are just going to drop it. 
# 
# ### Creating
# 
# We may want to create a new feature called Year based on Date to analyze the price change throughout the years.
# We may want to create a new feature called Month based on Date to analyze the price change throughout the months.

# <a id="ch3"></a>
# ## Exploratory Data Analysis
# <hr/>
# 
# ### Analyze by visualizing data
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.

# ### Pearson correlation matrix
# We use the Pearson correlation coefficient to examine the strength and direction of the linear relationship between two continuous variables.
# 
# The correlation coefficient can range in value from −1 to +1. The larger the absolute value of the coefficient, the stronger the relationship between the variables. For the Pearson correlation, an absolute value of 1 indicates a perfect linear relationship. A correlation close to 0 indicates no linear relationship between the variables. 
# 
# The sign of the coefficient indicates the direction of the relationship. If both variables tend to increase or decrease together, the coefficient is positive, and the line that represents the correlation slopes upward. If one variable tends to increase as the other decreases, the coefficient is negative, and the line that represents the correlation slopes downward.

# In[8]:


sns.set(style="whitegrid", font_scale=1)

plt.figure(figsize=(13,13))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
            annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})


# ### Price correlation
# * This allow us to explore labels that are highly correlated to the price.
# * sqft_living looks like a highly correlated label to the price, as well as grade, sqft_above, sqft_living15 and bathrooms.
# 
# ***Which features are more correlated to the price?***

# In[9]:


price_corr = df.corr()['price'].sort_values(ascending=False)
print(price_corr)


# ### Price feature 
# * Most of the house prices are between \\$0 and \\$1,500,000.
# * The average house price is \\$540,000.
# * Keep in mind that it may be a good idea to drop extreme values. For instance, we could focus on house from \\$0 to \\$3,000,000 and drop the other ones.
# * It seems that there is a positive linear relationship between the price and sqft_living.
# * An increase in living space generally corresponds to an increase in house price.

# In[10]:


f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.distplot(df['price'], ax=axes[0])
sns.scatterplot(x='price',y='sqft_living', data=df, ax=axes[1])
sns.despine(bottom=True, left=True)
axes[0].set(xlabel='Price in millions [USD]', ylabel='', title='Price Distribuition')
axes[1].set(xlabel='Price', ylabel='Sqft Living', title='Price vs Sqft Living')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()


# ### Bedrooms and floors box plots
# Box plot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot. Outliers may be plotted as individual points. The spacings between the different parts of the box indicate the degree of dispersion (spread). 
# 
# * We can see outliers plotted as individual points; this probably are the more expensive houses.
# * We can see that the price tends to go up when the house has more bedrooms. 

# In[11]:


sns.set(style="whitegrid", font_scale=1)

f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])
sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])
sns.despine(bottom=True, left=True)
axes[0].set(xlabel='Bedrooms', ylabel='Price', title='Bedrooms vs Price Box Plot')
axes[1].set(xlabel='Floors', ylabel='Price', title='Floors vs Price Box Plot')


# ### Waterfront, view and grade box plots
# * Waterfront houses tends to have a better price value.
# * The price of waterfront houses tends to be more disperse and the price of houses without waterfront tend to be more concentrated.
# * Grade and waterfront effect price. View seem to effect less but it also has an effect on price.

# In[12]:


f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])
sns.boxplot(x=df['view'],y=df['price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Waterfront', ylabel='Price', title='Waterfront vs Price Box Plot')
axes[1].set(xlabel='View', ylabel='Price', title='View vs Price Box Plot')

f, axe = plt.subplots(1, 1,figsize=(15,5))
sns.boxplot(x=df['grade'],y=df['price'], ax=axe)
sns.despine(left=True, bottom=True)
axe.set(xlabel='Grade', ylabel='Price', title='Grade vs Price Box Plot')


# <a id="ch4"></a>
# ## Working with Feature Data
# <hr/>
# 
# ### Correcting by dropping features
# By dropping features, we are dealing with fewer data points. Speeds up our notebook and eases the analysis. Based on our assumptions and decisions we want to drop the Id, zipcode and Date features. 

# In[13]:


df = df.drop('id', axis=1)
df = df.drop('zipcode',axis=1)


# <a id="ch5"></a>
# ## Feature engineering
# <hr/>
# 
# We want to engineer the date feature to make a year and month column. The feature date is as a string. With pd.to_datetime we can convert an argument to datetime.

# In[14]:


df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

df = df.drop('date',axis=1)

# Check the new columns
print(df.columns.values)


# ### House price trends
# * Looking the box plots we notice that there is not a big difference between 2014 and 2015.
# * The number of houses sold by month tends to be similar every month. 
# * The line plot show that around April there is an increase in house prices.

# In[15]:


f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x='year',y='price',data=df, ax=axes[0])
sns.boxplot(x='month',y='price',data=df, ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Year', ylabel='Price', title='Price by Year Box Plot')
axes[1].set(xlabel='Month', ylabel='Price', title='Price by Month Box Plot')

f, axe = plt.subplots(1, 1,figsize=(15,5))
df.groupby('month').mean()['price'].plot()
sns.despine(left=True, bottom=True)
axe.set(xlabel='Month', ylabel='Price', title='Price Trends')


# <a id="ch6"></a>
# ## Scaling and train test split
# <hr/>
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a mean squared error regression problem. We are also performing a category of machine learning which is called supervised learning as we are training our model with a given dataset.

# In[16]:


# Features
X = df.drop('price',axis=1)

# Label
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[17]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Normalizing / scaling the data
# We scale the feature data. To prevent data leakage from the test set, we only fit our scaler to the training set.

# In[18]:


scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# everything has been scaled between 1 and 0
print('Max: ',X_train.max())
print('Min: ', X_train.min())


# <a id="ch7"></a>
# ## Creating a model
# ***
# We estimate the number of neurons (units) from our features. Ex: X_train.shape (15117, 19). The optimizer is asking how you want to perform this gradient descent. In this case we are using the Adam optimizer and the mean square error loss function.

# In[19]:


model = Sequential()

# input layer
model.add(Dense(19,activation='relu'))

# hidden layers
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

# output layer
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# <a id="ch8"></a>
# ## Training the model
# ***
# Now that the model is ready, we can fit the model into the data.
# 
# Since the dataset is large, we are going to use batch_size. It is typical to use batches of the power of 2 (32, 64, 128, 256...). In this case we are using 128. The smaller the batch size, the longer is going to take.

# In[20]:


model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)


# ### Training loss per epoch
# * This plot shows the training loss per epoch.
# * This plot helps us to see if there is overfitting in the model. In this case there is no overfitting because both lines go down at the same time. 

# In[21]:


losses = pd.DataFrame(model.history.history)

plt.figure(figsize=(15,5))
sns.lineplot(data=losses,lw=3)
plt.xlabel('Epochs')
plt.ylabel('')
plt.title('Training Loss per Epoch')
sns.despine()


# <a id="ch9"></a>
# ## Evaluation on test data
# ---
# ### Regression Evaluation Metrics
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

# ### Predicting on brand new data
# In this part we are giving the model the test set to get a list of predictions. Then we compare the correct values with the list of predictions. We use different metrics to compare the predictions, in this case we use MAE, MSE, RMSE and Variance Regression Score. 
# 
# Let us start by analyzing the MAE, which is \\$103,500. This means that our model is off on average about \\$100,000.
# 
# ***Is that MAE good or bad?***
# 
# For that we must take into account our original data set and see what kind of values we have. For instance, the mean is 540,000, therefore the MEA is about 19% of the mean price. This is not a particularly good result.
# 
# To better understand this error, we can use the variance regression score, where the best possible score is 1.0 and lower values are worse. This tells you how much variance is being explain by your model. In our case we have 0.80 which is a normal result. 

# In[22]:


# predictions on the test set
predictions = model.predict(X_test)

print('MAE: ',mean_absolute_error(y_test,predictions))
print('MSE: ',mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)))
print('Variance Regression Score: ',explained_variance_score(y_test,predictions))

print('\n\nDescriptive Statistics:\n',df['price'].describe())


# ### Model predictions vs perfect fit
# * We can compare the model predictions with a perfect fit to see how accurate the model is.
# * The red line represents the perfect prediction. 
# * We are being punish by the outliers, which are the expensive houses. Our model is not good predicting luxury houses.
# * On the other hand, our model is good predicting the price of houses between o and \\$2 million. There is clearly a good fit. 
# * It may be worth it retraining our model just on price houses below \\$3 million.

# In[23]:


f, axes = plt.subplots(1, 2,figsize=(15,5))

# Our model predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')

errors = y_test.values.reshape(6484, 1) - predictions
sns.distplot(errors, ax=axes[0])

sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
axes[1].set(xlabel='Test True Y', ylabel='Model Predictions', title='Model Predictions vs Perfect Fit')


# <a id="ch10"></a>
# ## Predicting on a brand new house
# ---
# We are going to use the model to predict the price on a brand-new house. We are going to choose the first house of the data set and drop the price. single_house is going to have all the features that we need to predict the price. After that we need to reshape the variable and scale the features.
# 
# The original price is \\$221,900 and the model prediction is \\$280,000. 

# In[24]:


# fueatures of new house
single_house = df.drop('price',axis=1).iloc[0]
print(f'Features of new house:\n{single_house}')

# reshape the numpy array and scale the features
single_house = scaler.transform(single_house.values.reshape(-1, 19))

# run the model and get the price prediction
print('\nPrediction Price:',model.predict(single_house)[0,0])

# original price
print('\nOriginal Price:',df.iloc[0]['price'])


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Extract the required columns for training
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_model.predict(X_test)

# Print the predictions
print("Predictions:")
for i in range(len(y_pred)):
    print(f"Predicted price: {y_pred[i]}, Actual price: {y_test.iloc[i]}")

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_model.predict(X_test)

# Print the predictions
print("Predictions:")
for i in range(len(y_pred)):
    print(f"Predicted price: {y_pred[i]}, Actual price: {y_test.iloc[i]}")

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# ## References
# * [An Introduction to Statistical Learning with Applications in R](http://faculty.marshall.usc.edu/gareth-james/ISL/) - This book provides an introduction to statistical learning methods.
# * [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) - Use Python for Data Science and Machine Learning.

# **Thank you** for taking the time to read through my first exploration of a Kaggle dataset. I look forward to doing more!
# 
# If you have a question or feedback, do not hesitate to comment and if you like this kernel,<b><font color='green'> please upvote! </font></b>
