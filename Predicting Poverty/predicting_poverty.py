
get_ipython().magic('matplotlib inline')

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import cohen_kappa_score


# GETTING DATA READY 
DATA_DIR = os.path.join('C:', 'data\processed')

## CREATING DATA PATH 
data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}

# load training data
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
a_train.head()

# PLOTTING THE 'POOR' VS 'NON POOR' FOR COUNTRY A
a_train.poor.value_counts().plot.bar(title='Number of Poor for country A')
a_train.info()



## PRE-PROCESS DATA 
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


## GETTING TRAINING DATA READY 
print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)

## GETTING TEST DATA READY 
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)


## GETTING INDIVIDUAL LEVEL DATA READY 
data_paths_ind = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_indiv_train.csv'), 
                        'test':  os.path.join(DATA_DIR, 'A', 'A_indiv_test.csv')}, 
              
                  'B': {'train': os.path.join(DATA_DIR, 'B', 'B_indiv_train.csv'), 
                        'test':  os.path.join(DATA_DIR, 'B', 'B_indiv_test.csv')}, 
              
                  'C': {'train': os.path.join(DATA_DIR, 'C', 'C_indiv_train.csv'), 
                        'test':  os.path.join(DATA_DIR, 'C', 'C_indiv_test.csv')}}

# load training data
a_ind_train = pd.read_csv(data_paths_ind['A']['train'], index_col=['id','iid'])
b_ind_train = pd.read_csv(data_paths_ind['B']['train'], index_col=['id','iid'])
c_ind_train = pd.read_csv(data_paths_ind['C']['train'], index_col=['id','iid'])

### PREPROCESSING INDIVIDUAL LEVEL TRAINING DATA 
print("Country A")
aX_ind_train = pre_process_data(a_ind_train.drop('poor', axis=1))

print("\nCountry B")
bX_ind_train = pre_process_data(b_ind_train.drop('poor', axis=1))

print("\nCountry C")
cX_ind_train = pre_process_data(c_ind_train.drop('poor', axis=1))

### SUMMING OVER BY HOUSEHOLD ID 
aX_indGrp_train = aX_ind_train.groupby(['id']).sum()
bX_indGrp_train = bX_ind_train.groupby(['id']).sum()
cX_indGrp_train = cX_ind_train.groupby(['id']).sum()

### MERGING WITH HOUSEHOLD LEVEL TRAINING DATA 
aX_mrg_train = aX_train.merge(aX_indGrp_train, how='outer', left_index=True, right_index=True)
bX_mrg_train = bX_train.merge(bX_indGrp_train, how='outer', left_index=True, right_index=True)
cX_mrg_train = cX_train.merge(cX_indGrp_train, how='outer', left_index=True, right_index=True)

## GETTING INDIVIDUAL LEVEL TEST DATA READY 
a_ind_test = pd.read_csv(data_paths_ind['A']['test'], index_col=['id','iid'])
b_ind_test = pd.read_csv(data_paths_ind['B']['test'], index_col=['id','iid'])
c_ind_test = pd.read_csv(data_paths_ind['C']['test'], index_col=['id','iid'])

### PREPROCESSING INDIVIDUAL LEVEL TEST DATA 
a_ind_test = pre_process_data(a_ind_test, enforce_cols=aX_ind_train.columns)
b_ind_test = pre_process_data(b_ind_test, enforce_cols=bX_ind_train.columns)
c_ind_test = pre_process_data(c_ind_test, enforce_cols=cX_ind_train.columns)

### SUMMING OVER BY HOUSEHOLD ID 
a_indGrp_test = a_ind_test.groupby(['id']).sum()
b_indGrp_test = b_ind_test.groupby(['id']).sum()
c_indGrp_test = c_ind_test.groupby(['id']).sum()

### MERGING WITH HOUSEHOLD LEVEL TEST DATA 
a_mrg_test = a_test.merge(a_indGrp_test, how='outer', left_index=True, right_index=True)
b_mrg_test = b_test.merge(b_indGrp_test, how='outer', left_index=True, right_index=True)
c_mrg_test = c_test.merge(c_indGrp_test, how='outer', left_index=True, right_index=True)



# CASE A 
from sklearn.model_selection import train_test_split
aX_sp_train, aX_test, ay_sp_train, ay_test = train_test_split(aX_train, ay_train, random_state=42)
bX_sp_train, bX_test, by_sp_train, by_test = train_test_split(bX_train, by_train, random_state=42)
cX_sp_train, cX_test, cy_sp_train, cy_test = train_test_split(cX_train, cy_train, random_state=42)


# CASE B
aX_mrg_sp_train, aX_mrg_test, ay_mrg_sp_train, ay_mrg_test = train_test_split(aX_mrg_train, ay_train, random_state=42)
bX_mrg_sp_train, bX_mrg_test, by_mrg_sp_train, by_mrg_test = train_test_split(bX_mrg_train, by_train, random_state=42)
cX_mrg_sp_train, cX_mrg_test, cy_mrg_sp_train, cy_mrg_test = train_test_split(cX_mrg_train, cy_train, random_state=42)



## BUILDING RANDOM FOREST CLASSIFIER 
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


## FIT RANDOM FOREST MODEL     
model_a = train_model(aX_train, ay_train)
model_b = train_model(bX_train, by_train)
model_c = train_model(cX_train, cy_train)

a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)

# CONVERT PREDICTION to DATA FRAMES
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')
submission = pd.concat([a_sub, b_sub, c_sub])


## FIT RANDOM FOREST MODEL WITH MERGE TRAINING DATASET
model_a = train_model(aX_mrg_train, ay_train)
model_b = train_model(bX_mrg_train, by_train)
model_c = train_model(cX_mrg_train, cy_train)

a_pred = model_a.predict_proba(a_mrg_test)
b_pred = model_b.predict_proba(b_mrg_test)
c_pred = model_c.predict_proba(c_mrg_test)

# convert preds to data frames
a_merge_sub = make_country_sub(a_pred, a_mrg_test, 'A')
b_merge_sub = make_country_sub(b_pred, b_mrg_test, 'B')
c_merge_sub = make_country_sub(c_pred, c_mrg_test, 'C')

submission2 = pd.concat([a_merge_sub, b_merge_sub, c_merge_sub])
submission2.rename(columns={'country':'country_y', 'poor':'poor_y'}, inplace=True)
s2 = submission.join(submission2, sort=False)
s2.drop(['country_y', 'poor'], axis=1, inplace=True)
s2.rename(columns={'poor_y':'poor'}, inplace=True)


## FIT RANDOM FOREST MODEL with SPLIT DATA  
model_a = train_model(aX_sp_train, ay_sp_train)
model_b = train_model(bX_sp_train, by_sp_train)
model_c = train_model(cX_sp_train, cy_sp_train)

a_pred = model_a.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))


b_pred = model_b.predict(bX_test)
b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

c_pred = model_c.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


## FIT RANDOM FOREST MODEL with SPLIT MERGED DATA  
model_a = train_model(aX_mrg_sp_train, ay_mrg_sp_train)
model_b = train_model(bX_mrg_sp_train, by_mrg_sp_train)
model_c = train_model(cX_mrg_sp_train, cy_mrg_sp_train)

a_pred = model_a.predict(aX_mrg_test)
a_accuracy = accuracy_score(ay_mrg_test, a_pred)
print("Country A: ", a_accuracy)
print(metrics.classification_report(ay_mrg_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_mrg_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_mrg_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_mrg_test, a_pred))

b_pred = model_b.predict(bX_mrg_test)
b_accuracy = accuracy_score(by_mrg_test, b_pred)
print("Country B: ", b_accuracy)
print(metrics.classification_report(by_mrg_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_mrg_test, b_pred))
print("Log loss : ", metrics.log_loss(by_mrg_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_mrg_test, b_pred))

c_pred = model_c.predict(cX_mrg_test)
c_accuracy = accuracy_score(cy_mrg_test, c_pred)
print("Country C: ", c_accuracy)
print(metrics.classification_report(cy_mrg_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_mrg_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_mrg_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_mrg_test, c_pred))


## BUILDING LOGISTIC REGRESSION MODEL 
from sklearn.linear_model import LogisticRegression

### WITH DEFAULT OPTION AND ONLY HOUSEHOLD TRAINING DATA (PROVIDED)
logreg = LogisticRegression()

logreg.fit(aX_train, ay_train)
a_prob = logreg.predict_proba(a_test)
logreg.fit(bX_train, by_train)
b_prob = logreg.predict_proba(b_test)
logreg.fit(cX_train, cy_train)
c_prob = logreg.predict_proba(c_test)

a_sub=make_country_sub(a_prob, a_test, 'A')
b_sub= make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')
submission3 = pd.concat([a_sub, b_sub, c_sub])


## WITH BALANCED OPTION AND ONLY HOUSEHOLD TRAINING DATA (PROVIDED)
logreg = LogisticRegression(class_weight = 'balanced')

logreg.fit(aX_train, ay_train)
a_prob = logreg.predict_proba(a_test)
logreg.fit(bX_train, by_train)
b_prob = logreg.predict_proba(b_test)
logreg.fit(cX_train, cy_train)
c_prob = logreg.predict_proba(c_test)

a_sub=make_country_sub(a_prob, a_test, 'A')
b_sub= make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')
submission3_2 = pd.concat([a_sub, b_sub, c_sub])


### WITH DEFAULT OPTION AND USING MERGE HOUSEHOLD TRAINING DATA
logreg = LogisticRegression()

logreg.fit(aX_mrg_train, ay_train)
a_prob = logreg.predict_proba(a_mrg_test)
logreg.fit(bX_mrg_train, by_train)
b_prob = logreg.predict_proba(b_mrg_test)
logreg.fit(cX_mrg_train, cy_train)
c_prob = logreg.predict_proba(c_mrg_test)

a_sub=make_country_sub(a_prob, a_mrg_test, 'A')
b_sub= make_country_sub(b_prob, b_mrg_test, 'B')
c_sub = make_country_sub(c_prob, c_mrg_test, 'C')
submission4 = pd.concat([a_sub, b_sub, c_sub])


## WITH BALANCED OPTION AND USING MERGE HOUSEHOLD TRAINING DATA
logreg = LogisticRegression(class_weight='balanced')

logreg.fit(aX_mrg_train, ay_train)
a_prob = logreg.predict_proba(a_mrg_test)
logreg.fit(bX_mrg_train, by_train)
b_prob = logreg.predict_proba(b_mrg_test)
logreg.fit(cX_mrg_train, cy_train)
c_prob = logreg.predict_proba(c_mrg_test)

a_sub=make_country_sub(a_prob, a_mrg_test, 'A')
b_sub= make_country_sub(b_prob, b_mrg_test, 'B')
c_sub = make_country_sub(c_prob, c_mrg_test, 'C')
submission4_2 = pd.concat([a_sub, b_sub, c_sub])


logreg = LogisticRegression(class_weight = 'balanced')

logreg.fit(aX_sp_train, ay_sp_train)
a_prob = logreg.predict_proba(aX_test)
a_pred = logreg.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print ("Country A")
print("Accuracy of country A : ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

logreg.fit(bX_sp_train, by_sp_train)
b_prob = logreg.predict_proba(bX_test)
b_pred = logreg.predict(bX_test)
b_accuracy = accuracy_score(by_test, b_pred)
print ("Country B")
print("Accuracy of country B : ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))


logreg.fit(cX_sp_train, cy_sp_train)
c_prob = logreg.predict_proba(cX_test)
c_pred = logreg.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print ("Country C")
print("Accuracy of country C : ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))



logreg = LogisticRegression()

logreg.fit(aX_sp_train, ay_sp_train)
a_prob = logreg.predict_proba(aX_test)
a_pred = logreg.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print ("Country A")
print("Accuracy of country A : ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

logreg.fit(bX_sp_train, by_sp_train)
b_prob = logreg.predict_proba(bX_test)
b_pred = logreg.predict(bX_test)
b_accuracy = accuracy_score(by_test, b_pred)
print ("Country B")
print("Accuracy of country B : ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))


logreg.fit(cX_sp_train, cy_sp_train)
c_prob = logreg.predict_proba(cX_test)
c_pred = logreg.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print ("Country C")
print("Accuracy of country C : ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


## BUILD SUPPORT VECTOR MACHINE MODEL WITH CV=5
from sklearn.model_selection import cross_val_score
a_clf = svm.SVC(kernel='linear', C=1, probability=True)
a_scores = cross_val_score(a_clf, aX_train, ay_train, cv=5)
a_clf.fit(aX_train, ay_train)
print(a_scores)

b_clf = svm.SVC(kernel='linear', C=1, probability=True)
b_scores = cross_val_score(b_clf, bX_train, by_train, cv=5)
b_clf.fit(bX_train, by_train)
print(b_scores)

c_clf = svm.SVC(kernel='linear', C=1, probability=True)
c_scores = cross_val_score(c_clf, cX_train, cy_train, cv=5)
c_clf.fit(cX_train, cy_train)
print(c_scores)

a_pred = a_clf.predict_proba(a_test)
b_pred = b_clf.predict_proba(b_test)
c_pred = c_clf.predict_proba(c_test)

a_sub = make_country_sub(a_pred, a_test, 'A')
b_sub = make_country_sub(b_pred, b_test, 'B')
c_sub = make_country_sub(c_pred, c_test, 'C')
submission5 = pd.concat([a_sub, b_sub, c_sub])


## BUILD SUPPORT VECTOR MACHINE MODEL WITH CV=10
a_clf = svm.SVC(kernel='linear', C=1, probability=True)
a_scores = cross_val_score(a_clf, aX_train, ay_train, cv=10)
b_clf = svm.SVC(kernel='linear', C=1, probability=True)
b_scores = cross_val_score(b_clf, bX_train, by_train, cv=10)
c_clf = svm.SVC(kernel='linear', C=1, probability=True)
c_scores = cross_val_score(c_clf, cX_train, cy_train, cv=10)

a_clf.fit(aX_train, ay_train)
b_clf.fit(bX_train, by_train)
c_clf.fit(cX_train, cy_train)

a_pred = a_clf.predict_proba(a_test)
b_pred = b_clf.predict_proba(b_test)
c_pred = c_clf.predict_proba(c_test)

a_sub = make_country_sub(a_pred, a_test, 'A')
b_sub = make_country_sub(b_pred, b_test, 'B')
c_sub = make_country_sub(c_pred, c_test, 'C')
submission6 = pd.concat([a_sub, b_sub, c_sub])


## FIT SVM MODEL WITH CV =5 AND USING SPLIT TRAINING DATA ONLY 
clf = svm.SVC(kernel='linear', C=1, probability=True)
a_scores = cross_val_score(clf, aX_sp_train, ay_sp_train, cv=5)
clf.fit(aX_sp_train, ay_sp_train)
print(a_scores)
a_pred = clf.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

b_scores = cross_val_score(clf, bX_sp_train, by_sp_train, cv=5)
clf.fit(bX_sp_train, by_sp_train)
print(b_scores)
b_pred = clf.predict(bX_test)
b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

c_scores = cross_val_score(clf, cX_sp_train, cy_sp_train, cv=5)
clf.fit(cX_sp_train, cy_sp_train)
print(c_scores)
c_pred = clf.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))



## BUILDING GRID SEARCH MODEL USING TRAINING SPLIT DATA 
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# use grid search for tuning hyperparameters
svc = SVC(class_weight='balanced', probability=True)
params_space = {'kernel': ['linear', 'poly', 'rbf']}
# set cv to your K-fold cross-validation
gs = GridSearchCV(svc, params_space, n_jobs=-1, cv=5)

gs.fit(aX_train, ay_train)
a_prob = gs.predict_proba(a_test)

gs.fit(bX_train, by_train)
b_prob = gs.predict_proba(b_test)

gs.fit(cX_train, cy_train)
c_prob = gs.predict_proba(c_test)

a_sub = make_country_sub(a_prob, a_test, 'A')
b_sub = make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')
submission7 = pd.concat([a_sub, b_sub, c_sub])


## BUILDING GRID SEARCH MODEL USING TRAINING SPLIT DATA 
gs.fit(aX_sp_train, ay_sp_train)
a_pred = gs.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

gs.fit(bX_sp_train, by_sp_train)
b_pred = gs.predict(bX_test)
b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

gs.fit(cX_sp_train, cy_sp_train)
c_pred = gs.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


# Applying gridsearch method on merge dataset
gs.fit(aX_mrg_train, ay_train)
a_prob = gs.predict_proba(a_mrg_test)

gs.fit(bX_mrg_train, by_train)
b_prob = gs.predict_proba(b_mrg_test)

gs.fit(cX_mrg_train, cy_train)
c_prob = gs.predict_proba(c_mrg_test)

a_sub = make_country_sub(a_prob, a_mrg_test, 'A')
b_sub = make_country_sub(b_prob, b_mrg_test, 'B')
c_sub = make_country_sub(c_prob, c_mrg_test, 'C')
submission8 = pd.concat([a_sub, b_sub, c_sub])



## BUILDING ADABOOST CLASSIFIER MODEL 
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(model, aX_train, ay_train)
scores
model.fit(aX_train, ay_train)
a_prob = model.predict_proba(a_test)
model.fit(bX_train, by_train)
b_prob = model.predict_proba(b_test)
model.fit(cX_train, cy_train)
c_prob = model.predict_proba(c_test)

a_sub = make_country_sub(a_prob, a_test, 'A')
b_sub = make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')
submission11 = pd.concat([a_sub, b_sub, c_sub])



## BUILDING ADABOOST CLASSIFIER MODEL 
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)

model.fit(aX_sp_train, ay_sp_train)
a_pred = model.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

model.fit(bX_sp_train, by_sp_train)
b_pred = model.predict(bX_test)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

model.fit(cX_sp_train, cy_sp_train)
c_pred = model.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))



model = AdaBoostClassifier(n_estimators=100)

model.fit(aX_mrg_train, ay_train)
a_prob = model.predict_proba(a_mrg_test)
model.fit(bX_mrg_train, by_train)

b_prob = model.predict_proba(b_mrg_test)
model.fit(cX_mrg_train, cy_train)
c_prob = model.predict_proba(c_mrg_test)

a_sub = make_country_sub(a_prob, a_test, 'A')
b_sub = make_country_sub(b_prob, b_test, 'B')
c_sub = make_country_sub(c_prob, c_test, 'C')
submission12 = pd.concat([a_sub, b_sub, c_sub])



## BUILDING CLASS TREE MODEL 
from sklearn import tree
model = tree.DecisionTreeClassifier()

model = model.fit(aX_sp_train, ay_sp_train)
a_pred = model.predict(aX_test)
a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

model = model.fit(bX_sp_train, by_sp_train)
b_pred = model.predict(bX_test)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))


model = model.fit(cX_sp_train, cy_sp_train)
c_pred = model.predict(cX_test)
c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


## BUILDING K NEAREST NEIGHBOR MODEL
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
kmeans = KMeans(n_clusters=4)

train_scores = []
test_scores = []
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(aX_sp_train, ay_sp_train)
    train_score = knn.score(aX_sp_train, ay_sp_train)
    test_score = knn.score(aX_test, ay_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")

plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.savefig('knn1.png')
plt.show()


knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(aX_sp_train, ay_sp_train)
a_prob = knn.predict_proba(aX_test)
a_pred = knn.predict(aX_test)

a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))


train_scores = []
test_scores = []
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(bX_sp_train, by_sp_train)
    train_score = knn.score(bX_sp_train, by_sp_train)
    test_score = knn.score(bX_test, by_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")

plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.savefig('knn2.png')
plt.show()



knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(bX_sp_train, by_sp_train)
b_prob = knn.predict_proba(bX_test)
b_pred = knn.predict(bX_test)

b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_predict)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))


train_scores = []
test_scores = []
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(cX_sp_train, cy_sp_train)
    train_score = knn.score(cX_sp_train, cy_sp_train)
    test_score = knn.score(cX_test, cy_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")

plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.savefig('knn3.png')
plt.show()


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(cX_sp_train, cy_sp_train)
c_prob = knn.predict_proba(cX_test)
c_pred = knn.predict(cX_test)

c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


## BUILDING XGBOOST MODEL 
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

model = XGBClassifier()
model.fit(aX_sp_train, ay_sp_train)
a_pred = model.predict(aX_test)

a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

model.fit(bX_sp_train, by_sp_train)
b_pred = model.predict(bX_test)

b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

model.fit(cX_sp_train, cy_sp_train)
c_pred = model.predict(cX_test)

c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))


### SVM WITH ISOTONIC OPTION 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

clf = svm.SVC(kernel='linear', C=1, probability=True)
a_clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
a_clf_isotonic.fit(aX_sp_train, ay_sp_train)

a_prob_pos_isotonic = a_clf_isotonic.predict_proba(aX_test)[:, 1]
a_clf_isotonic_score = brier_score_loss(ay_test, a_prob_pos_isotonic)
print("With isotonic calibration: %1.3f" % a_clf_isotonic_score)


b_clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
b_clf_isotonic.fit(bX_sp_train, by_sp_train)

b_prob_pos_isotonic = b_clf_isotonic.predict_proba(bX_test)[:, 1]
b_clf_isotonic_score = brier_score_loss(by_test, b_prob_pos_isotonic)
print("With isotonic calibration: %1.3f" % b_clf_isotonic_score)

c_clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
c_clf_isotonic.fit(cX_sp_train, cy_sp_train)

c_prob_pos_isotonic = c_clf_isotonic.predict_proba(cX_test)[:, 1]
c_clf_isotonic_score = brier_score_loss(cy_test, c_prob_pos_isotonic)
print("With isotonic calibration: %1.3f" % c_clf_isotonic_score)

a_pred = a_clf_isotonic.predict(aX_test)
b_pred = b_clf_isotonic.predict(bX_test)
c_pred = c_clf_isotonic.predict(cX_test)

a_accuracy = accuracy_score(ay_test, a_pred)
print("Country A: ", a_accuracy)
a_roc = roc_auc_score(ay_test, a_pred)
print("ROC of country A : ", a_roc)
print(metrics.classification_report(ay_test, a_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(ay_test, a_pred))
print("Log loss : ", metrics.log_loss(ay_test, a_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(ay_test, a_pred))

b_accuracy = accuracy_score(by_test, b_pred)
print("Country B: ", b_accuracy)
b_roc = roc_auc_score(by_test, b_pred)
print("ROC of country B : ", b_roc)
print(metrics.classification_report(by_test, b_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(by_test, b_pred))
print("Log loss : ", metrics.log_loss(by_test, b_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(by_test, b_pred))

c_accuracy = accuracy_score(cy_test, c_pred)
print("Country C: ", c_accuracy)
c_roc = roc_auc_score(cy_test, c_pred)
print("ROC of country C : ", c_roc)
print(metrics.classification_report(cy_test, c_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(cy_test, c_pred))
print("Log loss : ", metrics.log_loss(cy_test, c_pred))
print("Cohen Kappa Score: ", cohen_kappa_score(cy_test, c_pred))

