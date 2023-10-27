# Импорт библиотек
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from langchain import LLMChain
import nltk
from sklearn.cluster import KMeans
import faiss


def clustering():
    nltk.download('wordnet')
    text = open('./knowledge_base_supporting/questions/knowledge_base.txt', encoding='utf-8').read()
    fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", text) if fragment.strip()]
    text = "".join(fragments)
    text = nltk.tokenize.sent_tokenize(text)  # Токенизация
    lemmatizer = nltk.WordNetLemmatizer()
    text = [" ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence)]) for sentence in text]

    # text = [word for word in text if word not in stopwords] # Удаление стоп-слов

    model = SentenceTransformer('all-MiniLM-L6-v2')

    text_vectors = []
    for sentence in text:
        vector = model.encode(sentence)
        text_vectors.append(vector)

    # clusters_count = find_optimal_clusters(text_vectors)
    # print("clusters_count: " + str(clusters_count))

    text_vectors = np.array(text_vectors)
    # Кластеризация
    num_clusters = 17  # Выбрано с помощью метода локтя
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(text_vectors)

    # Индексация
    index = faiss.IndexFlatL2(text_vectors.shape[1])
    index.add(text_vectors.astype('float32'))  # Убедитесь, что массив имеет тип данных float32

    # Поиск ответа
    question_vector = model.encode("Здравствуйте. Как пройти обучающие видео по программе?")
    question_array = np.array([question_vector]).astype('float32')  # Преобразование в двумерный массив NumPy

    cluster_id = kmeans.predict(question_array)[0]  # Теперь question_array должен быть двумерным массивом
    D, I = index.search(question_array, 10)
    print([text[i] for i in I[0] if kmeans.labels_[i] == cluster_id])


import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score


def find_optimal_clusters(data, max_clusters=84):
    scores = []
    for n_clusters in range(2, max_clusters):
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = model.fit_predict(data)
        score = calinski_harabasz_score(data, labels)
        scores.append(score)

    plt.plot(list(range(2, max_clusters)), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('CH Score')
    plt.show()

    return scores.index(max(scores)) + 2
