#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jovian


# In[2]:


from fbprophet import Prophet


# In[3]:


#install fbprophet 
#conda install -c conda-forge fbprophet


# In[4]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')
start=dt.datetime(2015,1,1)
end=dt.datetime.now()
df=web.DataReader("SHRIRAMCIT.NS",'yahoo',start,end)
print(df.head())


# In[5]:



df.to_csv('shriramcity.csv')


# In[10]:


# as the date is not an index so i have to reset the date column to make it as index
df=df.reset_index()


# In[11]:


#Let's rename the columns as required by fbprophet. Additioinally, fbprophet doesn't like the index to be a datetime...it wants to see 'ds' as a non-index column, so we won't set an index differnetly than the integer index
df[['ds','y']]=df[['Date','Adj Close']]


# In[13]:


#Now's a good time to take a look at your data. Plot the data using pandas' plot function
df.set_index('ds').y.plot()


# #Running Prophet
# Now, let's set prophet up to begin modeling our data using our promotions dataframe as part of the forecast
# 
# Note: Since we are using monthly data, you'll see a message from Prophet saying Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this. This is OK since we are workign with monthly data but you can disable it by using weekly_seasonality=True in the instantiation of Prophet.
# 
# I

# In[15]:


model = Prophet(weekly_seasonality=True)


# In[16]:


model.fit(df)


# We've instantiated the model, now we need to build some future dates to forecast into.

# In[17]:


future = model.make_future_dataframe(periods=365)
future.tail()


# 
# To forecast this future data, we need to run it through Prophet's model.

# In[21]:


forecast=model.predict(future)


# In[31]:


forecast.tail()


# In[22]:


model.plot(forecast)
plt.show()


# Additionally, prophet let's us take a at the components of our model, including the holidays. This component plot is an important plot as it lets you see the components of your model including the trend and seasonality

# In[23]:


# Python
fig2 = model.plot_components(forecast)


# Now that we have our model, let's take a look at how it compares to our actual values using a few different metrics - R-Squared and Mean Squared Error (MSE).
# 
# To do this, we need to build a combined dataframe with yhat from the forecasts and the original 'y' values from the data.
# 
# 

# In[20]:


metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()


# In[24]:


metric_df.tail()


# You can see from the above, that the last part of the dataframe has "NaN" for 'y'...that's fine because we are only concerend about checking the forecast values versus the actual values so we can drop these "NaN" values.

# In[25]:


metric_df.dropna(inplace=True)


# In[26]:


metric_df.tail()


# Now let's take a look at our R-Squared value

# In[28]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

r2_score(metric_df.y, metric_df.yhat)


# In[29]:


mean_squared_error(metric_df.y, metric_df.yhat)


# That's a large MSE value...and confirms my suspicion that this data is overfit and won't likely hold up well into the future. Remember...for MSE, closer to zero is better.
# 
# Now...let's see what the Mean Absolute Error (MAE) looks like.

# In[30]:


mean_absolute_error(metric_df.y, metric_df.yhat)


# In[33]:


jovian.commit()


# In[ ]:




