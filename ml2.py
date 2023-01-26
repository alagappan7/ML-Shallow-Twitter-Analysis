import numpy as np
import pandas as pd
#import BeautifulSoup
# utilities
import re
import nltk
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.svm import LinearSVC
from nltk.stem import wordnet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
snowball = SnowballStemmer(language='english')
word_lemma = WordNetLemmatizer()
# plotting
import seaborn as sns
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
# nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()
data = pd.read_csv("D:/NCL docs/Machine learning/coursework/Tweets_train.csv")
test = pd.read_csv("D:/NCL docs/Machine learning/coursework/Tweets_test.csv", encoding="latin-1")

#print(data.groupby('airline_sentiment').count())
"""ax = data.groupby('airline_sentiment').count().plot(kind='bar', title='Distribution of data',legend=False)
ax.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)
# Storing data in lists.

"""
#sns.countplot(x='airline_sentiment', data=data, palette = "Set2")
text, sentiment = list(data['text']), list(data['airline_sentiment'])
#NLP - Stopword removal
test_text, test_sentiment = list(test['text']),list(test['airline_sentiment'])
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
nlp = [] 
pun=[]  
url=[]
lem =[]
#stop_words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
def NLP(text):
  
  for sent in text:
    redef1 = ""
    wordlist = sent.split()
    for word in wordlist:
      if (word.lower() not in ENGLISH_STOP_WORDS):
        if redef1=="":
          redef1=word
        else:  
          redef1 = redef1 + " " + word

        """print(redef1)
        print("length of old data:", len(text))
        print("length of new data:", len(redef1))"""
    
    nlp.append(redef1)
# Removing Punctuation  
def PUNC(nlp):
  for rem in nlp:
    redef2=" "
    clean2 = re.sub("[^A-Za-z ]", "" , rem)
    pun.append(clean2)

def URL(pun):
  for content in pun:
    clean3 = re.sub('http\S+', '', content) #replaces string which does not contains any white space characters and starts with http
    url.append(clean3)

#Tokenisation Lemma Stem
def Gram(url):
  for sentence in url: #token
    #tokens = word_tokenize(sentence)
    onesent = " "
    #for eachword in sentence:
    stemmed = snowball.stem(sentence.lower())
    lemmatized = word_lemma.lemmatize(stemmed)
    #  onesent = onesent + " " + lemmatized
    lem.append(lemmatized)

    


NLP(text)
PUNC(nlp)
nlp.clear()
URL(pun)
pun.clear()
Gram(url)
url.clear()
model_Data = data[['text','airline_sentiment']]
#print("ype is",type(lem))
#print(lem)
data['protext']=lem
dataset = [data['protext'],data['airline_sentiment']]
X = data.protext
y = data.airline_sentiment
lem.clear()
NLP(test_text)
PUNC(nlp)
nlp.clear()
URL(pun)
pun.clear()
Gram(url)
url.clear()


test['protext']=lem
test_data = test.protext

print(test_data)
test_target = test.airline_sentiment

vectoriser = TfidfVectorizer()
vectoriser.fit(X)
X = vectoriser.transform(X)
z = vectoriser.transform(test_data)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))
lem.clear()

from sklearn.metrics import roc_curve, auc
SVCmodel = LinearSVC()
SVCmodel.fit(X, y)

y_pred = SVCmodel.predict(z)

print("test results",accuracy_score(y_pred, test_target)*100)

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
print(classification_report(test_target , y_pred))


#naive bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X,y)
# Predict the categories of the test data
predicted_categories = model.predict(z)
print("test results",accuracy_score(test_target, predicted_categories)*100)
print(classification_report(test_target , predicted_categories))

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X,y)
predictions = dtree.predict(z)
from sklearn.metrics import classification_report,confusion_matrix
print("decision tree")
print(classification_report(test_target,predictions))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
#X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.20)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
rfc_pred = rfc.predict(z)
#print("CONFUSSION MATRIX:")
#print(confusion_matrix(y_test,rfc_pred))
print("ACCURACY REPORT rand:")
print(classification_report(test_target,rfc_pred))
