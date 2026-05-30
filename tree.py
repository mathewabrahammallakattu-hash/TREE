import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = pd.read_csv('Decision_Tree.csv')


label_encoder = LabelEncoder()

data["temp_encoded"]=label_encoder.fit_transform(data["Temperature"])
data["vibe_encoded"]=label_encoder.fit_transform(data["Vibration"])
data["fail_encoded"]=label_encoder.fit_transform(data["Failure"])

X = data[["temp_encoded", "vibe_encoded"]]
Y = data["fail_encoded"]
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, Y)
input = [[0, 2]] 
prediction = model.predict(input)
print("Prediction:", prediction[0])
if prediction == 1:
  print("Failure")
else:
  print("No Failure")