import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob.tokenizers import WordTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem.porter import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker
from textblob import Word, TextBlob
from tqdm import tqdm, tqdm_pandas

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv('100_data.csv')

import string
remove = string.punctuation
remove = remove.replace("-", "") # don't remove hyphens
pattern = r"[{}]".format(remove)
lemmatizer = WordNetLemmatizer()

def remove_punctuation(transcript):
    sentences = sent_tokenize(str(transcript))
    sent=[]
    for sen in sentences:
        # sen = re.sub('[^\w\s]','', sen)
        # sent.append(sen)
        sentence = sen.translate(str.maketrans('','', string.punctuation))
        sent.append(sentence)
    transcript = ' '.join(sent)
    return transcript

def sentence_correction(transcript):
    transcript = TextBlob(transcript)
    transcript = transcript.correct()
    print(transcript)
    return transcript

def lemmatized_transcript(transcript):
    words = word_tokenize(str(transcript))
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    lemmatized_transcript = ' '.join(words)
    return lemmatized_transcript

def sentences_stemming(transcript):
    words = word_tokenize(str(transcript))
    porter_stemmer = PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    stemmed_transcripts = ' '.join(words)
    return stemmed_transcripts

# def stopWordRemoval(transcripts, stop_words = set()):
#     words = word_tokenize(str(transcript))
#     filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#     transcripts = ' '.join(filtered_sentence)
#     return transcripts

from spellchecker import SpellChecker
spell = SpellChecker()
def sentence_correction_spellchecker(transcript):
    words = word_tokenize(str(transcript))
    corrected_words = []
    words = [spell.correction(word) for word in words]
    spelling_corrected = ' '.join(str(word) for word in words)
    return spelling_corrected

columns = ['transcript','category']
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    transcript = row['transcript']
    transcript = remove_punctuation(transcript)
    transcript = sentence_correction_spellchecker(transcript)
    transcript = lemmatized_transcript(transcript)
    row['transcript'] = transcript

print('================================================================')
print(df.category.value_counts())
print('================================================================')

df['label'] = df.category.map({
    'computer science' : 0,
    'biology' : 1,
    'environmental studies' : 2
})

test_df = df
shuffled = test_df.sample(frac=1)

X = shuffled.transcript
y = shuffled.label

import os
data_dirs = ['./data', './data/y/', './data/cv/', './data/tfidf/']
for dir in data_dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y, shuffle=True)

print('================================================================')
print("Entries in X_train: {} \n Entries in X_test: {} ".format(X_train.count(),X_test.count()))
print('================================================================')
print('================================================================')
y_test.value_counts()
print('================================================================')
y_train.to_csv('./data/y/train.csv', index=False)
y_test.to_csv('./data/y/test.csv', index=False)

cv = CountVectorizer(ngram_range=(1,3))
X_trainCV = cv.fit_transform(X_train)
X_trainDF = pd.DataFrame(X_trainCV.toarray(), columns = cv.get_feature_names_out())
X_trainDF.to_csv('./data/cv/train.csv', index=False)

X_testCV = cv.transform(X_test)
X_testDF = pd.DataFrame(X_testCV.toarray(), columns = cv.get_feature_names_out())
X_testDF.to_csv('./data/cv/test.csv', index=False)

tvec = TfidfVectorizer(ngram_range=(1,3))
X_train_tv = tvec.fit_transform(X_train)
X_trainTDF = pd.DataFrame(X_train_tv.toarray(), columns = tvec.get_feature_names_out())
X_trainTDF.to_csv('./data/tfidf/train.csv', index=False)

X_test_tv = tvec.transform(X_test)
X_testTDF = pd.DataFrame(X_test_tv.toarray(), columns = tvec.get_feature_names_out())
X_testTDF.to_csv('./data/tfidf/test.csv', index=False)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# labels = ['CS', 'Bio', 'ES']

# Log_Reg = LogisticRegression(max_iter=1000)
# Log_Reg.fit(X_trainDF, y_train)
# lr_cv_pred = Log_Reg.predict(X_testDF)
# print('================================================================')
# print('LogisticRegression (CV)')
# print(classification_report(y_test,lr_cv_pred, target_names=labels))
# print('================================================================')


# Log_Reg_tv = LogisticRegression()
# Log_Reg_tv.fit(X_trainTDF, y_train)
# lr_tf_pred = Log_Reg_tv.predict(X_testTDF)
# print('================================================================')
# print('LogisticRegression (Tf-idf)')
# print(classification_report(y_test,lr_cv_pred, target_names=labels))
# print('================================================================')