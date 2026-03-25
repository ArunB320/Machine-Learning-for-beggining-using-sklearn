import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#to load the data
df = pd.read_csv("/content/drive/MyDrive/spam (1).csv")
df.head()

#convert text into vector
def text_to_vector(text):
  if(text == "ham"):
    return 1
  else:
    return 0

#apply  the function
df['Category'] = df['Category'].apply(text_to_vector)
y = df['Category']

#convert mssg into vector
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])

#divide data test and training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#select model
model = RandomForestClassifier(n_estimators=100, max_depth=15,class_weight='balanced')

#train model
model.fit(X_train,y_train)

#make prediction of test data
y_pred = model.predict(X_test)

#check model accuracy
cf_score = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

#print model score
print(f"classification report: {cf_score}")
print(f"accuracy score: {accuracy_score}")
print(cross_val_score(model, X, y, cv=5).mean())

#predict the new data
def predict_text(text):
  vectorize = vectorizer.transform([text])[0]
  prediction = model.predict(vectorize)

  if(prediction == 1):
    return "ham"
  else:
    return "spam"

message = "2 days left for this pizza party! if you would to join the party click the link!"
print(predict_text(message))
