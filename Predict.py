import pandas as pd

#Model aus Datei lesen
clf = pd.read_pickle(r'classifier_object.pickle')

data = pd.read_excel('data/Predict_Daten.xls')

x = data.loc[:, data.columns != 'class']

y_pred = clf.predict(x)

print(y_pred)