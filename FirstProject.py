
# coding: utf-8

# In[1]:


# Import the pandas library.
import pandas as pd
# Read in the data.
games = pd.read_csv("board_games.csv")
# Print the names of the columns in games.
print games.columns


# In[2]:


print games.shape


# In[3]:


import matplotlib.pyplot as plt
plt.hist(games["average_rating"])
plt.show()


# In[18]:


plt.hist(games["average_weight"])
plt.show()


# In[4]:


'''Exploring the 0 ratings
'''
games[games["average_rating"] == 0]


# In[5]:


print (games[games["average_rating"] == 0].iloc[0])


# In[6]:


print (games[games["average_rating"] > 0].iloc[0])


# In[7]:


# Remove any rows without user reviews.
games = games[games["users_rated"] > 0]
# Remove any rows with missing values.
games = games.dropna(axis = 0)


# In[8]:


# Import the kmeans clustering model.
from sklearn.cluster import KMeans

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_
print labels


# In[9]:


from sklearn.decomposition import PCA
#create a PCA model
pca_2 = PCA(2)
#fit the PCA model on the numeric columns from earlier
plot_columns = pca_2.fit_transform(good_columns)
#make a scatter plot of each game, shaded according to 
#cluster assignment
plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = labels)
#show the plot
plt.show()


# In[10]:


games.corr()["average_rating"]


# In[51]:


#get all the columns from the dataframe
columns = games.columns.tolist()
print columns ;print
#filter the columns to remove ones we don't want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]
print columns
#score the variable we'll be predicting on
target = "average_rating"


# In[52]:


#import a convenience function to split the sets 
from sklearn.cross_validation import train_test_split
#generate the training set, set random_state to be able to replicate results
train = games.sample(frac = 0.8, random_state= 1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
#print the shapes of both sets
print(train.shape)
print(test.shape)


# In[55]:


# Import the linearregression model.
from sklearn.linear_model import LinearRegression
# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns],train[target])


# In[56]:


from sklearn.metrics import mean_squared_error
predictions = model.predict(test[columns])
mean_squared_error(predictions, test[target])

