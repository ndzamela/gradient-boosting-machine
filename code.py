# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")



# Explore the data 


# mean and standard deviation of their age


# Display the statistics of age for each gender of all the races (race feature).


# encoding the categorical features.


# Split the data and apply decision tree classifier


# Perform the boosting task
 

#  plot a bar plot of the model's top 10 features with it's feature importance score


#  Plot the training and testing error vs. number of trees



