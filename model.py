import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
data=data.iloc[:10000,:]


stopwords_set= set(stopwords.words('english'))
emoji_pattern=re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
def preprocessing(clean_text):
    clean_text=re.sub('<[^>]*>', '',clean_text)
    emojis=emoji_pattern.findall(clean_text)
    clean_text=re.sub('[\W+]', ' ',clean_text.lower()) + ' '.join(emojis).replace('-', '')

    prter= PorterStemmer()
    clean_text=[prter.stem(word) for word in clean_text.split() if word not in stopwords_set]

    return " ".join(clean_text)
data['clean_text'] = data['clean_text'].fillna('')
data['clean_text'] = data['clean_text'].apply(preprocessing)
posdata = data[data['category'] == 1]['clean_text']
negdata = data[data['category'] == -1]['clean_text']
neudata = data[data['category'] == 0]['clean_text']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,use_idf=True,norm='12',smooth_idf=True)
y=data.category.values
x=tfidf.fit_transform(data.clean_text)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)


from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
y_pred=clf.predict(X_test)


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


import pickle
pickle.dump(clf,open('clf.pkl','wb'))
pickle.dump(tfidf,open('tfidf.pkl','wb'))


def prediction(comment):
    preprocessed_comment=preprocessing(comment)
    comment_list=[preprocessed_comment]
    comment_vector=tfidf.transform(comment_list)
    prediction=clf.predict(comment_vector)[0]
    return prediction
prediction_result=prediction('') 




if prediction_result == 1:
    print("pos com")
elif prediction_result == 0:
    print("neu com")
else:
    print("neg com")