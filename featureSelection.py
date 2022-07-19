# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
#from matplotlib import pyplot
# load data
names = ['gender','StageID','SectionID','Semester', 'Relation','raisedhands','VisITedResources','AnnouncementsView','Discussion','ParentAnsweringSurvey',
         'ParentschoolSatisfaction','StudentAbsenceDays','NationalITy_Egypt','NationalITy_Iran','NationalITy_Iraq','NationalITy_Jordan',
         'NationalITy_KW','NationalITy_Lybia','NationalITy_Morocco','NationalITy_Palestine','NationalITy_SaudiArabia',
         'NationalITy_Syria','NationalITy_Tunis','NationalITy_USA','NationalITy_lebanon','NationalITy_venzuela','PlaceofBirth_Egypt'
         ,'PlaceofBirth_Iran','PlaceofBirth_Iraq','PlaceofBirth_Jordan','PlaceofBirth_KuwaIT','PlaceofBirth_Lybia','PlaceofBirth_Morocco',
         'PlaceofBirth_Palestine','PlaceofBirth_SaudiArabia','PlaceofBirth_Syria','PlaceofBirth_Tunis','PlaceofBirth_USA','PlaceofBirth_lebanon',
         'PlaceofBirth_venzuela','GradeID_G-02','GradeID_G-04','GradeID_G-05','GradeID_G-06','GradeID_G-07',
         'GradeID_G-08','GradeID_G-09','GradeID_G-10','GradeID_G-11','GradeID_G-12','Topic_Arabic','Topic_Biology','Topic_Chemistry','Topic_English',
         'Topic_French','Topic_Geology','Topic_History','Topic_IT','Topic_Math','Topic_Quran','Topic_Science','Topic_Spanish','Class']

dataframe = pd.read_csv('encoded_file.csv', names=names,skiprows=1)

array = dataframe.iloc[:,:].values

X= array[:,0:62]
Y = array[:,62]
#chi2
#test = SelectKBest(score_func=chi2, k=7)
#fit = test.fit(X, Y)
#X_new=test.fit_transform(X, Y)

#for i in range(len(fit.scores_)):
#	print('Feature %d: %f' % (i, fit.scores_[i]))
    
#mutualinfo
test = SelectKBest(score_func=mutual_info_classif, k=7)
fit = test.fit(X, Y)
X_new=test.fit_transform(X, Y)

for i in range(len(fit.scores_)):
	print('Feature %d: %f' % (i, fit.scores_[i]))
    

#pyplot.bar([i for i in range(len(fit.scores_))], fit.scores_)
#pyplot.show()


#extra tree classifier
#model = ExtraTreesClassifier(n_estimators=10)
#model.fit(X, Y)
#print(model.feature_importances_)



#logistic regression
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression


#model = LogisticRegression(solver='lbfgs')
#rfe = RFE(model, 7)
#fit = rfe.fit(X, Y)
#print("Num Features: %d" % fit.n_features_)
#print("Selected Features: %s" % fit.support_)
#print("Feature Ranking: %s" % fit.ranking_)