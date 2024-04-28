from .embeddings import Embeddings
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, PER, NamesExtractor, DatesExtractor, MoneyExtractor, AddrExtractor, Doc)
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from catboost import CatBoostClassifier

answer_class = pd.read_csv('./data/answer_class.csv')

device = torch.device('cuda:0')

embeddings = Embeddings('intfloat/multilingual-e5-large', device)


def get_embs(df):
    emb = embeddings.exec(df['Question'])
    X = pd.DataFrame(emb.numpy())
    return X

nltk.download('stopwords')
stop_words = set(stopwords.words("russian"))

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def lematizate_sentance(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    new_sent = ""

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma not in stop_words:
            new_sent += f"{token.lemma} "
    return new_sent

def lematizate_sentances(texts):
    new_texts = []
    for text in texts:
        new_texts.append(lematizate_sentance(text))
    return new_texts

"""Фунции для полноценного фунционирования"""

#Технические заполнители, дальше будут удалены с помощью drop
def tech_fill(df):
    df['Category'] = [None] * len(df)
    df['answer_class'] = [None] * len(df)
    df['index'] = [None] * len(df)
    return df

def data_prep(texts, tfidf_vectorizer):
    #Перевод списка вопросов к нужному формату
    train_data = pd.DataFrame([])
    train_data['Question'] = texts
    train_data = tech_fill(train_data)

    #добавление TF-IDF
    documents = lematizate_sentances(texts)
    tfidf_matrix = tfidf_vectorizer.transform(documents)
    tfidf_table = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    train_data = pd.concat([train_data,tfidf_table], axis=1)

    return train_data

def stages_pred(X, model_cl_category, model_cl_answer, verbose=0, proba=False):
    X_= get_embs(X)
    X = pd.concat([X,X_], axis = 1)
    X = X.drop(['Question', 'Category', 'answer_class', 'index'], axis= 1 )
    pred1 = model_cl_category.predict(X)

    # if verbose == 1:
    #     print(f"Категория {pred1}")

    df = pd.DataFrame([])
    df['Category'] = pred1[:,0]
    X = pd.concat([df, X], axis=1)
    X['Category'] = X['Category'].astype('category')

    pred2 = model_cl_answer.predict(X) if not proba else model_cl_answer.predict_proba(X)
    return pred2

def pipeline_predict(text,model_cl_category, model_cl_answer, tfidf_vectorizer):
  #предсказание
  answer = stages_pred(data_prep(text, tfidf_vectorizer),model_cl_category, model_cl_answer, verbose=1)
  list_answers = []
  for a in answer:
        list_answers.append(answer_class.iloc[a]['Answer'].values)

  return list_answers

def pipeline_predict2(text, model_cl_category, model_cl_answer, tfidf_vectorizer):
    #предсказание
    answer = stages_pred(data_prep(text, tfidf_vectorizer), model_cl_category, model_cl_answer, verbose=1, proba=True)
    answer_new = sorted([(answer[0][i], i) for i in range(len(answer[0]))], key=lambda x: x[0], reverse=True)[:3]
    list_answer = [answer_class.iloc[a[1]]['Answer'] for a in answer_new]
    return list_answer



model_cl_category = CatBoostClassifier()
model_cl_category.load_model('./models/model_cl_category.bin')
model_cl_answer = CatBoostClassifier()
model_cl_answer.load_model('./models/model_cl_answer.bin')

# Загрузка TfidfVectorizer из файла
tfidf_vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")



text = ['когда я залутаю диплом?', 'во сколько встреча?']

answers = pipeline_predict(text, model_cl_category, model_cl_answer, tfidf_vectorizer)


"""-----------------------------"""

def f_data_prep2(texts):
    #Перевод списка вопросов к нужному формату
    train_data = pd.DataFrame([])
    train_data['Question']=texts

    return train_data

def f_pred2(texts, model):
    df = f_data_prep2(texts)
    emb = embeddings.exec(df['Question'])
    X = pd.DataFrame(emb.numpy())
    return model.predict_proba(X)

model_filter_model = CatBoostClassifier()
model_filter_model.load_model('./models/filter_model.bin')


def filter_question(text, model_filter_model):
    pred = f_pred2(text, model_filter_model)
    tf_pred = pred > 0.3
    return tf_pred

f_pred2(['Когда появится расписание?'], model_filter_model)