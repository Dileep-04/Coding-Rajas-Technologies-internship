from flask import Flask, request, render_template

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

app = Flask(__name__)
stopwords_set= set(stopwords.words('english'))
emoji_pattern=re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
def preprocessing(clean_text):
    clean_text=re.sub('<[^>]*>', '',clean_text)
    emojis=emoji_pattern.findall(clean_text)
    clean_text=re.sub('[\W+]', ' ',clean_text.lower()) + ' '.join(emojis).replace('-', '')

    prter= PorterStemmer()
    clean_text=[prter.stem(word) for word in clean_text.split() if word not in stopwords_set]

    return " ".join(clean_text)

@app.route('/')
def index():
     return render_template("project1sa.html")

@app.route('/predict', methods=['POST', 'GET'])

def predict():
     if request.method == 'POST':
          comment=request.form['text']
          cleaned_comment=preprocessing(comment)
          comment_vector=tfidf.transform([cleaned_comment])
          prediction=clf.predict(comment_vector)[0]

          return render_template('project1sa.html',prediction = prediction)

if __name__=='__main__':
     app.run(debug=True)
     