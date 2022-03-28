import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def distance(x,y):
    return np.sqrt(((x-y)**2).sum())

def knn(query,x,y,k=9):
    dist = []
    for i in range(x.shape[0]):
        d = distance(query,x[i])
        dist.append([d,y[i]])
    dk = sorted(dist,key=lambda x:x[0])[:k]
    val = [x[1] for x in dk]
    new_vals = np.unique(val,return_counts=True)
    index = new_vals[1].argmax()
    prediction = new_vals[0][index]
    return prediction



cancer_data = pd.read_csv("D:\yadav\DownloadsD\Prostate_Cancer.csv")
cancer_data.drop(['id'],axis=1,inplace=True)
cancer_data['diagnosis_result'] = [1 if i == 'M' else 0 for i in cancer_data['diagnosis_result']]
y = cancer_data['diagnosis_result']
x_data = cancer_data.drop('diagnosis_result',axis=1)
x_data = x_data.values
y = y.values


scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform(x_data)


x_T,x_t,y_T,y_t = train_test_split(x,y,test_size=0.2,random_state=5)


prediction = []
for i in range(x_t.shape[0]):
    prediction.append(knn(x_t[i], x_T, y_T))


print(int(accuracy_score(y_t,prediction)*100))
