#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:13:01 2019

@author: jericolinux
"""

### Initialization
# Import Libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Feature engineering Libraries
import re

# Import Machine Learning Reqs
from sklearn import preprocessing

# Import Machine Learning Libraries
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import xgboost as xgb


app = Flask(__name__)

# =============================================================================
# Data Preprocessing
# =============================================================================
# Non-Optimized Preprocessing
grouped_median_train = pd.read_csv("preprocess_gmt.csv")
test = pd.read_csv("prefinal_sample.csv")
gbm =  pickle.load(open('model.pkl', 'rb'))
clf =  pickle.load(open('feature_reducer.pkl', 'rb'))
testcols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','Embarked']
coltypes = [[0, 0,'0', '0', 0.0, 0, 0, '0', 0.0, '0', '0']]

def preprocess(combined):

    # Dropping Columns that aren't needed
    combined = combined.drop(columns='PassengerId')

    # Reverse Pclass
    combined['Pclass_reversed'] = 4 - combined['Pclass']

    # Data creation: Title in exchange for Names
    pattern = r" ([A-Za-z]+)\."
    idx = combined.index
    title_list = []

    name_idx = 0
    for col in combined:
        if (col == 'Name'):
            break
        name_idx += 1

    # Browse through each df_copy rows
    # And get the title of each names
    for row in range(len(combined)):
        dataset = combined.iloc[row, name_idx]
        searched = re.search(pattern, dataset)
        title_list.append(searched[0])

    title_series = pd.Series(title_list, idx)

    # Change Name and its values to Title series
    combined['Name'] = title_series
    combined = combined.rename(columns={"Name": "Title"})\

    # Create dictionary
    title_dictionary = {
        " Capt.": "Officer",
        " Col.": "Officer",
        " Major.": "Officer",
        " Jonkheer.": "Royalty",
        " Don.": "Royalty",
        " Sir." : "Royalty",
        " Dr.": "Officer",
        " Rev.": "Officer",
        " Countess.":"Royalty",
        " Mme.": "Mrs.",
        " Mlle.": "Miss.",
        " Ms.": "Miss.",
        " Mr." : "Mr.",
        " Mrs." : "Mrs.",
        " Miss." : "Miss.",
        " Master." : "Master.",
        " Lady." : "Royalty",
        " Dona.": "Royalty"
    }

    # a map of more aggregated title
    # we map each title
    combined['Title'] = combined['Title'].map(title_dictionary)

    def fill_age(row):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) & 
            (grouped_median_train['Title'] == row['Title']) & 
            (grouped_median_train['Pclass_reversed'] == row['Pclass_reversed'])
        ) 
        return grouped_median_train[condition]['Age'].values[0]

    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    # LabelEncode the Sex
    le = preprocessing.LabelEncoder()
    combined.at[:,'Sex']= le.fit_transform(combined['Sex'])

    # Create FamSize Feature
    combined['FamSize'] = combined['SibSp'] + combined['Parch']

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamSize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamSize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamSize'].map(lambda s: 1 if 5 <= s else 0)

    # Data creation: Ticket Titles in exchange for Tickets
    pattern = r"([A-Za-z0-9/\.]+ )"
    idx = combined.index
    title_list = []

    ticket_idx = 0
    for colname in combined:
        if (colname == 'Ticket'):
            break
        ticket_idx += 1


    # Browse through each df_copy rows
    # And get the title of each names
    for row in range(len(combined)):
        dataset = combined.iloc[row, ticket_idx]
        searched = re.search(pattern, str(dataset))
        if searched:
            title_list.append(searched[0][0])
        else:
            title_list.append("NormalTicket")

    title_series = pd.Series(title_list, idx)

    # Change Name and its values to Title series
    combined['Ticket'] = title_series
    combined = combined.rename(columns={"Ticket": "Ticket Title"})

    combined.at[combined['Fare'].isna(), 'Fare'] = combined['Fare'].median()

    # Data correction for Cabin
    # Data creation: Title in exchange for Cabin
    pattern = r"([A-Za-z]+)"
    idx = combined.index
    title_list = []

    cabin_idx = 0
    for col in combined:
        if (col == 'Cabin'):
            break
        cabin_idx += 1

    # Browse through each df_copy rows
    # And get the title of each names
    for row in range(len(combined)):
        dataset = combined.iloc[row, cabin_idx]
        try: 
            if np.isnan(dataset):
                title_list.append('U')
        except:
            searched = re.search(pattern, dataset)
            title_list.append(searched[0][0])

    title_series = pd.Series(title_list, idx)

    # # Change Cabin and its values to Cabin first word series
    combined['Cabin'] = title_series

    # One-Hot Encoding for Title
    title_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, title_dummies], axis=1)
    combined = combined.drop(columns='Title')

    # One Hot Encoding for Cabin
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined = combined.drop(columns='Cabin')

    # One Hot Encoding for Embarked
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined = combined.drop(columns='Embarked')

    # One Hot Encoding for Ticket Title
    tick_title_dummies = pd.get_dummies(combined['Ticket Title'], prefix='Ticket_Title')
    combined = pd.concat([combined, tick_title_dummies], axis=1)
    combined = combined.drop(columns='Ticket Title')

    # Recover Train and test data set
    test2 = combined

    # Get the equivalent instance with the whole feature
    new_instance = pd.DataFrame(np.array([[0]*len(test.columns)]), columns=test.columns)
    for col in test2.columns:
        if col in test.columns:
            new_instance.at[0,col] = test2[col]
    
    return new_instance

def predictor(new_instance):
    # Avoid overfitting by reducing Features
    model = SelectFromModel(clf, prefit=True)
    # train_reduced = model.transform(train)
    test_reduced = model.transform(new_instance)

    return gbm.predict(test_reduced)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = []
    features = [x for x in request.form.values()]
    new_instance = pd.DataFrame(coltypes, columns=testcols)
    i = 0;
    j = 0;
    total_str = '';
    for val in features:
        try:
            new_instance.iat[0,i] = int(val)
        except:
#            total_str = total_str + '=====' + str(type(new_instance.iloc[0,i])) + ': ' + str(val)
            new_instance.iat[0,i] = str(val)
        i += 1
        
#        total_str = total_str + '\n' + str(type(val)) + ': ' + str(val)
    for cols in new_instance:
        total_str = total_str + '=====' + ': ' + str(new_instance[cols])
    
    new = preprocess(new_instance)
    prediction = predictor(new)
    
    name = features[2]
    if prediction:
        return render_template('index.html', prediction_text=name+' will survive!')
    else:
        return render_template('index.html', prediction_text=name+' will die!')

#    return render_template('index.html', prediction_text=str(new[0]) + ' ' + str(new[1]))    
        
#    for cols in new_instance:
#        new_instance.loc[0,cols]
    
#    return render_template('index.html', prediction_text=total_str)

    

if __name__ == "__main__":
    app.run(debug=True)