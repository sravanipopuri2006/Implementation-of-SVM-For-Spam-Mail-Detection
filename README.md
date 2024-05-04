# EXP-09-Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import chardet
2.ead the dataset
3.Import SVC from sklearn
4.Fit the data in the model and run the algorithm
```
## Program:
```

/*
Program to implement the SVM For Spam Mail Detection.
Developed by: POPURI SRAVANI
RegisterNumber:  212223320117
*/
import chardet
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv("spam.csv", encoding='Windows=1252')

data.head()

data.info()

data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)   

```

## Output:
## RESULT
![Screenshot 2024-05-04 234753](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/2a878fb9-a7c6-47b5-ad00-f495b591d7a6)
## data.head()
![Screenshot 2024-05-04 234805](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/af2b91d4-29ee-4d1e-96e1-5341a7c00050)
## data.info()
![Screenshot 2024-05-04 234816](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/2c6e3b30-53f5-4b36-ab23-7e502bcc2abf)
## data.isnull.sum()
![Screenshot 2024-05-04 234825](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/9593e69d-7f0c-498e-be01-3b353e218b97)
## y_pred
![Screenshot 2024-05-04 234833](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/fb075f96-b766-4087-8bb2-6d88cc3d7104)
## Acuuracy
![Screenshot 2024-05-04 234840](https://github.com/sravanipopuri2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139778301/c708e4e3-fc7f-41a9-bfe9-48ac488cc873)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
