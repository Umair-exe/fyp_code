import numpy as np
import pandas as pd


dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:,:].values;

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth','GradeID','Topic','Semester','Relation','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays','Class'] 

df = pd.get_dummies(dataset, columns = categorical_cols)
Y = df.values

#ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0,1,2,4,6])],remainder='passthrough')
#Y = ct.fit_transform(dataset)


labelencoder_x = LabelEncoder()

Y[:,0] = labelencoder_x.fit_transform(X[:,3])
Y[:,1] = labelencoder_x.fit_transform(X[:,5])


df['StageID'] = df['StageID'].replace(X[:,3],Y[:,0])
df['SectionID'] = df['SectionID'].replace(X[:,5],Y[:,1])

df.to_csv('encoded_file.csv')










                      
                