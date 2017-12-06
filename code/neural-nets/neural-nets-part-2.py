
# coding: utf-8

# # Setup

# In[1]:


import keras
import pandas as pd
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from numpy import *
import sys
# sys.path.append(r"../../data")
inputs = '../../data/train.csv'


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

all_data_df = pd.read_csv(inputs)
real_test = pd.read_csv('../../data/test.csv')


# In[3]:


frames = [all_data_df, real_test]
all_data_df = pd.concat(frames)


# In[4]:


def preprocess_data(df):
    comparison_set = {1, 0}
    ultimate_df = df.copy(deep=True)
    ultimate_df = pd.get_dummies(ultimate_df)
    for column in ultimate_df:
        temp_set = set(ultimate_df[column].unique())
        is_not_boolean = (temp_set!=comparison_set)
        if(is_not_boolean and column!='SalePrice'):
            mean = ultimate_df[column].mean()
            std = ultimate_df[column].std()
            ultimate_df[column] = (ultimate_df[column] - mean) / std
    return ultimate_df

all_data_processed_df = preprocess_data(all_data_df)
all_data_processed_df.shape


# In[5]:


where_is_nan = isnan(all_data_processed_df['SalePrice'])
real_test_df = all_data_processed_df[where_is_nan]


# In[6]:


real_train = all_data_processed_df[~where_is_nan]


# In[7]:


train, test = train_test_split(real_train, test_size=0.3)

def separate_x_y(new_df):
    y = new_df['SalePrice']
    new_df = new_df.drop('SalePrice', axis=1)
    return new_df, y

x_train, y_train = separate_x_y(train)
x_test, y_test = separate_x_y(test)


# In[8]:


real_test_df, dummy_test = separate_x_y(real_test_df)


# In[9]:


x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_test = x_test.as_matrix()
y_test = y_test.as_matrix()
real_test = real_test_df.as_matrix()


# In[10]:


from numpy import *

where_are_NaNs = isnan(x_train)


# In[11]:


x_train[where_are_NaNs] = 0
where_are_NaNs = isnan(x_test)
x_test[where_are_NaNs] = 0

# x_test = x_test_n
# y_test = y_test_n


# ### 2 hidden layers 8622

# In[12]:


model = Sequential()

# hidden layer 1
# hidden layer with 64 nodes
num_hidden_layer_1_units = 120
num_hidden_layer_2_units = 60
num_hidden_layer_3_units = 30

model.add(Dense(units=num_hidden_layer_1_units, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.7))

# hidden layer 2
model.add(Dense(units=num_hidden_layer_2_units, input_dim=num_hidden_layer_1_units, activation='relu'))
model.add(Dropout(0.7))

# # hidden layer 3
model.add(Dense(units=num_hidden_layer_3_units, input_dim=num_hidden_layer_2_units, activation='relu'))
model.add(Dropout(0.7))

# output layer
model.add(Dense(units=1, input_dim = num_hidden_layer_3_units, activation='relu'))

model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=0.0001))

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=10, verbose=0,
                                              mode='auto')


# In[13]:


model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_test, y_test))


# In[14]:


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(loss_and_metrics)
print('finished')


# In[15]:


real_test.shape


# In[16]:


prediction = model.predict(real_test)


# In[17]:


real_test_df['prediction'] = prediction


# In[18]:


real_train2 =  pd.read_csv(inputs)


# In[19]:


id_iterator = range(real_train2.shape[0] + 1, (real_train2.shape[0] + real_test_df.shape[0] + 1 ))


# In[20]:



ids = list(id_iterator)


# In[21]:


real_test_df['id'] = list(id_iterator)


# In[22]:


submission = real_test_df[['id', 'prediction']]


# In[23]:


submission.rename(columns={'prediction': 'SalePrice'}, inplace=True)


# In[24]:


submission.to_csv('submission_kaggle_9.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




