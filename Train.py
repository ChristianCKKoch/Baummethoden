#Import Pandas library
import pandas as pd
#import method from sklearn to split our data into training and test data
from sklearn.model_selection import train_test_split
#import DecisionTreeClassifier from the Sklearn library
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle as pi

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data/data_banknote_authentication.csv', names=attribute_names)

#Shuffle data
data = data.sample(frac=1)

#Shows the first 5 rows of the data
data.head()

#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

#Create a classifier object 
classifier = DecisionTreeClassifier() 

#Classfier builds Decision Tree with training data
classifier = classifier.fit(x_train, y_train) 

#Shows importances of the attributes according to our model 
classifier.feature_importances_

#Save in file
clf_file = "classifier_object.pickle"
f = open(clf_file, 'wb')
pi.dump(classifier, f)
f.close()