import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("training_model/survey lung cancer.csv")
#print(df.head())
#print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
#print(df.shape)


encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
#df.head()
    
cat_feats = ['GENDER']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
#print(final_data.head())

X=final_data.drop(['LUNG_CANCER'],axis=1)
y=final_data['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



scaler=StandardScaler()
X_train['AGE']=scaler.fit_transform(X_train[['AGE']])
X_test['AGE']=scaler.transform(X_test[['AGE']])
#X_train.head()


#model = RandomForestClassifier(n_estimators=600)
#model.fit(X_train,y_train)
#prediction_test = model.predict(X_test)
#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test,prediction_test))


param_grid={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
rcv=RandomizedSearchCV(SVC(),param_grid,cv=5)
rcv.fit(X_train,y_train)
y_pred_svc=rcv.predict(X_test)
confusion_svc=confusion_matrix(y_test,rcv.predict(X_test))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_svc,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
#print(classification_report(y_test,y_pred_svc))
#print(f'\nBest Parameters of SVC model is : {rcv.best_params_}\n')

model = SVC(gamma=10,C=100)
model.fit(X_train,y_train)
y_pred_svc=model.predict(X_test)
confusion_svc=confusion_matrix(y_test,y_pred_svc)
plt.figure(figsize=(8,8))
sns.heatmap(confusion_svc,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_svc))

print(final_data.head())

import pickle
pickle.dump(rcv, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[69,1,2,2,1,1,2,1,2,2,2,2,2,2,1]]))

