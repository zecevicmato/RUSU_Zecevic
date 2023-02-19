from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix,classification_report

stroke_data = pd.read_csv("C:/Users/MATO/PythonProjects/deploy-lr/ml-model/dataset_stroke.csv")

X = stroke_data.drop(columns = 'stroke', axis = 1)
Y = stroke_data['stroke']



# Undersampling
rus = RandomUnderSampler(random_state=2)
X_resampled, Y_resampled = rus.fit_resample(X, Y)

# Podjela na skup za treniranje i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=2)


trees = DecisionTreeClassifier(max_depth=5)
trees.fit(X_resampled, Y_resampled)
pickle.dump(trees, open('dec_trees.pkl', 'wb'))

trees.fit(X_train, y_train)


y_train_pred = trees.predict(X_train)
y_test_pred = trees.predict(X_test)


print("Accuracy on Training data for Decision Trees:", accuracy_score(y_train, y_train_pred))
print("Accuracy on Test data for Decision Trees:", accuracy_score(y_test, y_test_pred))

print("Confusion matrix: ")
confusion_matrix = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix)
print()
print("Classification report: ")
report = classification_report(y_test, y_test_pred)
print(report)