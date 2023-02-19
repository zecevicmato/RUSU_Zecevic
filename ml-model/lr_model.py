from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import pickle


stroke_data = pd.read_csv("C:/Users/MATO/PythonProjects/deploy-lr/ml-model/dataset_stroke.csv")

X = stroke_data.drop(columns = 'stroke', axis = 1)
Y = stroke_data['stroke']


# Undersampling
rus = RandomUnderSampler(random_state=2)
X_resampled, Y_resampled = rus.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.2, stratify = Y_resampled, random_state = 2)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_resampled, Y_resampled)
pickle.dump(lr_model, open('lr_model.pkl', 'wb'))

X_train_prediction = lr_model.predict(X_resampled)
training_data_accuracy = accuracy_score(X_train_prediction, Y_resampled)
print('Accuracy on Training data for Logistic Regression: ', training_data_accuracy)

X_test_prediction = lr_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data for Logistic Regression: ', test_data_accuracy)

print("Confusion matrix: ")
confusion_matrix = confusion_matrix(Y_test, X_test_prediction)
print(confusion_matrix)
print()
print("Classification report: ")
report = classification_report(Y_test, X_test_prediction)
print(report)
