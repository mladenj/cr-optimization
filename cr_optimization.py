import sys

# data analysis and wrangling
import pandas as pd
import numpy as np
import pickle

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


N_JOBS = 6
CV = 5
SCORING_METHOD = 'f1'
RANDOM_STATE = 42
TEST_MODE = True

print(sys.argv)

if len(sys.argv) > 1:
    TEST_MODE = False
    filepath = sys.argv[1]
else:
    filepath = '../data/campaign_821471.log'

print('test mode: ', TEST_MODE)
print('filepath:', filepath)

#####################
# Data Loading
#####################

df = pd.read_csv(filepath, sep='\t')

imbalance_weight = round(len(df)/len(df[df['label'] == 1]),2)

print('Number of Impressions: ', len(df))
print('Number of Conversions: ', len(df[df['label'] == 1]))
print('----------------------')
print('Percentage: ', round(1/imbalance_weight,4))
print('Imbalance weight: ', imbalance_weight)

#####################
# Data Wrangling
#####################

df_model = df.copy()
print('before dropping NaN:',df_model.shape)
df_model.dropna(inplace=True)
print('after dropping NaN:', df_model.shape)

# Transform 'request_tld' column
at_least_one = df_model[df_model.label == 1].groupby('request_tld').count().index
df_model['request_tld'] = df_model['request_tld'].apply(lambda x: x if x in at_least_one else 'Others')

# Simplify Hour
df_model['hour'] = df_model['hour'].apply(lambda x: int(x / 4))

# Create Weekend
df_model['day_of_week'] = df_model['day_of_week'].apply(lambda x: 0 if x <= 5 else 1)

# Column selection - keep only the ones that will be used/converted for the final model
fixed_drop_columns = ['advertiser_id','campaign_id','country_code','content_category_ids','url_category_ids','project_id']
temp_drop_columns = ['url_category_ids', 'content_category_ids', 'location_id','organization' ] #'request_tld'
sel_drop_columns = [] #'platform','network',organization','browser_ver'

drop_columns = fixed_drop_columns + temp_drop_columns + sel_drop_columns

df_model.drop(drop_columns, axis=1, inplace=True)
print('df_model.shape: ',df_model.shape)

# Treating categorical data
num_cols = ['install_week','dma']
cat_cols = ['creative_id','keyword_id','state','browser_ver','platform','network','ad_blocker','hour','day_of_week','request_tld']

cat_to_num = df_model[cat_cols].select_dtypes(include=[np.number]).columns
df_model[cat_to_num] = df_model[cat_to_num].applymap(str)

# One Hot Encoding for all columns with categorical data
df_model = pd.concat([df_model[['label'] + num_cols], pd.get_dummies(df_model[cat_cols], drop_first=True)], axis=1)


#####################
# Data Preparation
#####################

# Undersampled for testing purposes
num_conv = len(df_model[df_model['label'] == 1])
df_model_us = pd.concat([df_model[df_model['label'] == 1],df_model[df_model['label'] == 0].sample(n=num_conv)])

if TEST_MODE:
    # reduced dataset
    X = df_model_us.iloc[:, 1:]
    y = df_model_us.iloc[:, 0]
else:
    # full dataset
    X = df_model.iloc[:, 1:]
    y = df_model.iloc[:, 0]

# Scaling numerical columns
scaler = StandardScaler().fit(df_model.loc[:,num_cols])
df_model.loc[:,num_cols] = scaler.transform(df_model.loc[:,num_cols])

# Forming train/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2, stratify=y)
print('train/test datasets: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#############################################
############ Modeling / Training ############
#############################################

print('----------------------------')
print('Modeling / Training')
print('----------------------------')

train_scores = {}
test_scores = {}
test_probas = {}
test_predictions = {}
model_names = []
models = {}

######################
# Logistic Regression
######################

model_name = 'LogReg'

print('---------------------')
print(model_name)
print('---------------------')

model_names.append(model_name)

mlmodel = LogisticRegression()

# Setting hyperparameters
C = [0.01,0.1,1]
class_weight = ['balanced',{0:1, 1:imbalance_weight/3},{0:1, 1:imbalance_weight/2},{0:1, 1:imbalance_weight}]
hyperparameters = dict(C=C, class_weight=class_weight)

cvmodel = GridSearchCV(mlmodel, hyperparameters, cv=CV, verbose=10, scoring=SCORING_METHOD, return_train_score=False)
cvmodel.fit(X_train,y_train)

# predict validation(test) set
y_test_pred = cvmodel.predict(X_test)

# saving results/models
models[model_name] = cvmodel
test_scores[model_name] = f1_score(y_test, y_test_pred)

########################
# Random Forest
########################

model_name = 'Random Forest'

print('---------------------')
print(model_name)
print('---------------------')

model_names.append(model_name)

mlmodel = RandomForestClassifier()

# Setting hyperparameters
class_weight = ['balanced', {0:1, 1:imbalance_weight/2},{0:1, 1:imbalance_weight/3}]
hyperparameters = dict(class_weight=class_weight, n_estimators = [20], max_depth=[20])

cvmodel = GridSearchCV(mlmodel, hyperparameters, cv=CV, verbose=10, scoring=SCORING_METHOD, return_train_score=False)
cvmodel.fit(X_train,y_train)

# predict validation(test) set
y_test_pred = cvmodel.predict(X_test)

# saving results/models
models[model_name] = cvmodel
test_scores[model_name] = f1_score(y_test, y_test_pred)

########################
# Naive Bayes
########################

model_name = 'NB'

print('---------------------')
print(model_name)
print('---------------------')

model_names.append(model_name)

mlmodel = BernoulliNB()

# Setting hyperparameters
cp_range = np.round(np.arange(0.0, 1.1, 0.1),2)
class_prior = list(zip(cp_range,reversed(cp_range)))
hyperparameters = dict(class_prior=class_prior)

cvmodel = GridSearchCV(mlmodel, hyperparameters, cv=CV, verbose=10, scoring=SCORING_METHOD, return_train_score=False)
cvmodel.fit(X_train,y_train)

# predict validation(test) set
y_test_pred = cvmodel.predict(X_test)

# saving results/models
models[model_name] = cvmodel
test_scores[model_name] = f1_score(y_test, y_test_pred)

################
# RESULTS
###############

print('---------------------')
print('Results: ', test_scores)
print('---------------------')
best_model = max(test_scores, key=test_scores.get)
print('best model: ', best_model)
print(SCORING_METHOD, ' score:', test_scores[best_model])

# save model to disk
filename = model_name + '_model.pkl'
pickle.dump(models[best_model], open(filename, 'wb'))
print('model saved to: ' + filename)