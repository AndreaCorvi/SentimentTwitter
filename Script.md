# SentimentTwitter
# Sentiment Analysis on a Twitter Dataset, found on Kaggle. Used Python and Naive Bayes
# Author: Corvi Andrea

### Libraries

import re
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from IPython.display import Image as im
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
from nltk.corpus import stopwords
from nltk import ngrams
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_curve, precision_score
import sklearn.metrics as metrics

### Importing data

data = pd.read_csv(r'C:\Users\corvi\Desktop\TM Project\train1.csv')

### Data Cleaning

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

data['clean_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*") 
data.head()

## EDA
data.head(3)
data[data['label'] == 1].head(10)
data[data['label'] == 0].head(10)
data.shape
data["label"].value_counts()
length_data = data['tweet'].str.len()
plt.hist(length_data, bins=20, label="Tweets")
plt.legend()
plt.show()

##  Stopwords
data.clean_tweet = [w for w in data.clean_tweet if w.lower() not in stopwords.words('english')]
## Stemmer
ps = nltk.PorterStemmer()
data.clean_tweet = [ps.stem(l) for l in data.clean_tweet]
## Data Splitting
X = data.clean_tweet
y = data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
train1=pd.concat([X_train,y_train], axis=1)
test1=pd.concat([X_test,y_test], axis=1)
# Tokenization
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train1.clean_tweet)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
# Train
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train1.label)
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(train1.clean_tweet, train1.label)  
# Test
y_pred = text_clf.predict(test1.clean_tweet)
# Classification Evaluation Measures
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

# Plot of the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ax= plt.subplot()
df_cm = pd.DataFrame(cm, index = [i for i in "AB"],columns = [i for i in "AB"])
plt.figure( figsize=(320,320) )
sns.heatmap(df_cm, annot=True, ax = ax, cmap='Blues', fmt='g'); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
plt.show()

### Roc Curve and AUC

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
roc_curve(y_test, y_pred)
plt.figure(figsize=(32,10))
plt.plot(fpr, tpr, marker='.')
print("Area under the curve=" + str(roc_auc))

#### Support Vector Machine
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(train1.clean_tweet, train1.label)  

y_pred = text_clf.predict(test1.clean_tweet)
np.mean(predicted == twenty_test.target)         

### Wordclouds

mask = np.array(Image.open(r'C:\Users\corvi\Desktop\TM Project\twitter.jpg'))

# Negative

nwords = ' '.join([text for text in data['clean_tweet'][data['label'] == 1]])
wcn = WordCloud(width=800, height=400,background_color="white", max_words=2000, mask=mask).generate(nwords)
plt.figure( figsize=(50,40) )
plt.imshow(wcn, interpolation='bilinear')
plt.axis("off")
plt.show()

# Positive

pwords = ' '.join([text for text in data['clean_tweet'][data['label'] == 0]])
wcp = WordCloud(width=800, height=400,background_color="white", max_words=2000, mask=mask).generate(pwords)
plt.figure( figsize=(50,40) )
plt.imshow(wcp, interpolation='bilinear')
plt.axis("off")
plt.show()

## Main Hashtags

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# Pos

HT_regular = hashtag_extract(data['clean_tweet'][data['label'] == 0])
HT_regular = sum(HT_regular,[])
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})     
d = d.nlargest(columns="Count", n = 15) 
plt.figure(figsize=(32,10))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# Neg

HT_negative = hashtag_extract(data['clean_tweet'][data['label'] == 1])
HT_negative = sum(HT_negative,[])    
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
e = e.nlargest(columns="Count", n = 15)   
plt.figure(figsize=(32,10))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
