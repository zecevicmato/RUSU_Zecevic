import numpy as np
import pandas as pd

datasetInfo = pd.read_csv('C:/Users/MATO/PythonProjects/deploy-lr/ml-model/dataset_stroke.csv')

print("Broj redaka i stupaca: ")
print(datasetInfo.shape)
print("Informacije o podacima: ")
datasetInfo.info()
print()


print("Provjera postoje li vrijednosti koje nedostaju: ")
print(datasetInfo.isnull().sum())
print()

print("Distribucija ciljne varijable: ")
print(datasetInfo['stroke'].value_counts())
print()