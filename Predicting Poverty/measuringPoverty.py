
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[2]:


# data directory
DATA_DIR = os.path.join('C:', 'data\processed')


# In[3]:


data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}


# In[4]:


# load training data
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')


# In[5]:


a_train.head()


# In[6]:


a_train.poor.value_counts().plot.bar(title='Number of Poor for country A')


# In[7]:


a_train.info()


# In[8]:


# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
        

    df = standardize(df)
    print("After standardization {}".format(df.shape))
        
    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    
    return df


# In[9]:


print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)


# In[10]:


from sklearn.ensemble import RandomForestClassifier

def train_model(features, labels, **kwargs):
    
    # instantiate model
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    
    # train model
    model.fit(features, labels)
    
    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print(f"In-sample accuracy: {accuracy:0.2%}")
    
    return model


# In[13]:


model_a = train_model(aX_train, ay_train)
model_b = train_model(bX_train, by_train)
model_c = train_model(cX_train, cy_train)


# In[14]:


a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')


# In[15]:


a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)


# In[16]:


a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)


# In[17]:


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# In[18]:


# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')


# In[19]:


submission = pd.concat([a_sub, b_sub, c_sub])


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


# In[43]:


from sklearn.model_selection import cross_val_score
a_clf = svm.SVC(kernel='linear', C=1, probability=True)
a_scores = cross_val_score(a_clf, aX_train, ay_train, cv=5)


# In[45]:


a_scores


# In[47]:


a_clf.fit(aX_train, ay_train)


# In[22]:


a_scores


# In[49]:


a_pred = a_clf.predict_proba(a_test)


# In[50]:


b_clf = svm.SVC(kernel='linear', C=1, probability=True)
b_scores = cross_val_score(b_clf, bX_train, by_train, cv=5)


# In[51]:


c_clf = svm.SVC(kernel='linear', C=1, probability=True)
c_scores = cross_val_score(c_clf, cX_train, cy_train, cv=5)


# In[55]:


b_clf.fit(bX_train, by_train)
c_clf.fit(cX_train, cy_train)


# In[56]:


b_pred = b_clf.predict_proba(b_test)
c_pred = c_clf.predict_proba(c_test)


# In[57]:


a_sub = make_country_sub(a_pred, a_test, 'A')
b_sub = make_country_sub(b_pred, b_test, 'B')
c_sub = make_country_sub(c_pred, c_test, 'C')


# In[60]:


submission5 = pd.concat([a_sub, b_sub, c_sub])
submission5.rename(columns={'country':'country_y', 'poor':'poor_y'}, inplace=True)
s5 = submission.join(submission5, sort=False)


# In[ ]:


s5.drop(['country_y', 'poor'], axis=1, inplace=True)
s5.rename(columns={'poor_y':'poor'}, inplace=True)


# In[65]:


a_clf = svm.SVC(kernel='linear', C=1, probability=True)
a_scores = cross_val_score(a_clf, aX_train, ay_train, cv=10)
b_clf = svm.SVC(kernel='linear', C=1, probability=True)
b_scores = cross_val_score(b_clf, bX_train, by_train, cv=10)
c_clf = svm.SVC(kernel='linear', C=1, probability=True)
c_scores = cross_val_score(c_clf, cX_train, cy_train, cv=10)


# In[66]:


a_clf.fit(aX_train, ay_train)
b_clf.fit(bX_train, by_train)
c_clf.fit(cX_train, cy_train)


# In[67]:


a_pred = a_clf.predict_proba(a_test)
b_pred = b_clf.predict_proba(b_test)
c_pred = c_clf.predict_proba(c_test)


# In[68]:


a_sub = make_country_sub(a_pred, a_test, 'A')
b_sub = make_country_sub(b_pred, b_test, 'B')
c_sub = make_country_sub(c_pred, c_test, 'C')


# In[69]:


submission6 = pd.concat([a_sub, b_sub, c_sub])
submission6.rename(columns={'country':'country_y', 'poor':'poor_y'}, inplace=True)
s6 = submission.join(submission6, sort=False)


# In[70]:


s6.drop(['country_y', 'poor'], axis=1, inplace=True)
s6.rename(columns={'poor_y':'poor'}, inplace=True)


# In[72]:


s6.head()


# In[73]:


s6.to_csv('submission6.csv')


# In[75]:


from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# unbalanced classification
#X, y = make_classification(n_samples=1000, weights=[0.1, 0.9])

# use grid search for tuning hyperparameters
svc = SVC(class_weight='balanced', probability=True)
params_space = {'kernel': ['linear', 'poly', 'rbf']}
# set cv to your K-fold cross-validation
gs = GridSearchCV(svc, params_space, n_jobs=-1, cv=5)
# fit the estimator


# In[76]:


gs.fit(aX_train, ay_train)
a_prob = gs.predict_proba(a_test)


# In[77]:


gs.fit(bX_train, by_train)
b_prob = gs.predict_proba(b_test)


# In[78]:


gs.fit(cX_train, cy_train)
c_prob = gs.predict_proba(c_test)


# In[79]:


a_sub = make_country_sub(a_prob, a_test, 'A')
b_sub = make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')


# In[80]:


submission7 = pd.concat([a_sub, b_sub, c_sub])
submission7.rename(columns={'country':'country_y', 'poor':'poor_y'}, inplace=True)
s7 = submission.join(submission7, sort=False)


# In[81]:


s7.drop(['country_y', 'poor'], axis=1, inplace=True)
s7.rename(columns={'poor_y':'poor'}, inplace=True)


# In[82]:


s7.head()


# In[83]:


s7.to_csv('submission7.csv')


# In[28]:


from sklearn.model_selection import cross_val_predict
from sklearn import datasets, linear_model


# In[29]:


lasso = linear_model.Lasso()


# In[30]:


a_pred = cross_val_predict(lasso, aX_train, ay_train)


# In[32]:


len(a_pred)


# In[33]:


cross_val_predict(a_clf, aX_train, ay_train, cv=10)


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import (StratifiedKFold, cross_val_score,
                                      train_test_split)


# In[36]:


cv = StratifiedKFold(ay_train, 10)
logreg = LogisticRegression()


# In[37]:


proba = cross_val_predict(logreg, aX_train, ay_train, cv=cv, method='predict_proba')


# In[41]:


logreg.fit(aX_train, ay_train)
a_prob = logreg.predict_proba(a_test)


# In[42]:


a_prob


# In[71]:


from sklearn.metrics import log_loss

