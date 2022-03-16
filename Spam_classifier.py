# importing data set from drive
import pandas as pd

df = pd.read_csv('D:/Pradeep/CDAC/DATA SETS/smsspamcollection/SMSSpamCollection', sep= '\t', names=["label","messege"])


#data cleaning and data preprocessing

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0, len(df)):
    Sent_review = re.sub('[^a-zA-Z]', ' ', df['messege'][i])
    Sent_review = Sent_review.lower()
    Sent_review = Sent_review.split()
    
    Sent_review = [ps.stem(word) for word in Sent_review if not word in stopwords.words('English')]
    Sent_review = ' '.join(Sent_review)
    corpus.append(Sent_review)
    
 # bag of words created below
from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(max_features=1000)

X = countvec.fit_transform(corpus).toarray()

Y = pd.get_dummies(df['label']) 
Y = Y.iloc[:,1].values

#Model Training

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size = 0.20, random_state = 0)

# naive bayes Model used as classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_prediction = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test, y_prediction) 

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_prediction)
