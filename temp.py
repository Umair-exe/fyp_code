import numpy as np
import pandas as pd


dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:,:].values;

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


categorical_cols = [ 'NationalITy', 'PlaceofBirth','GradeID','Topic'] 

df = pd.get_dummies(dataset, columns = categorical_cols)
Y = df.values

#ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0,1,2,4,6])],remainder='passthrough')
#Y = ct.fit_transform(dataset)

# 'gender','Semester','Relation','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays'
labelencoder_x = LabelEncoder()

Y[:,0] = labelencoder_x.fit_transform(X[:,0]) #gender
Y[:,1] = labelencoder_x.fit_transform(X[:,3]) #stageid
Y[:,4] = labelencoder_x.fit_transform(X[:,8]) #relation
Y[:,9] = labelencoder_x.fit_transform(X[:,13]) #parentanswering
Y[:,10] = labelencoder_x.fit_transform(X[:,14]) #parentssatisfaction
Y[:,11] = labelencoder_x.fit_transform(X[:,15]) #studentabsence
Y[:,12] = labelencoder_x.fit_transform(X[:,16]) #class


Y[:,2] = labelencoder_x.fit_transform(X[:,5]) #sectionid
Y[:,3] = labelencoder_x.fit_transform(X[:,7]) #semester 



df['gender'] = df['gender'].replace(X[:,0],Y[:,0])
df['StageID'] = df['StageID'].replace(X[:,3],Y[:,1])

df['Relation'] = df['Relation'].replace(X[:,8],Y[:,4])
df['ParentAnsweringSurvey'] = df['ParentAnsweringSurvey'].replace(X[:,13],Y[:,9])
df['ParentschoolSatisfaction'] = df['ParentschoolSatisfaction'].replace(X[:,14],Y[:,10])
df['StudentAbsenceDays'] = df['StudentAbsenceDays'].replace(X[:,15],Y[:,11])
df['Class'] = df['Class'].replace(X[:,16],Y[:,12])
df['SectionID'] = df['SectionID'].replace(X[:,5],Y[:,2])
df['Semester'] = df['Semester'].replace(X[:,7],Y[:,3])

df.to_csv('encoded_file.csv')










                      
                