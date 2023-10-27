import csv
import math
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter

from nltk import FreqDist
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter
import nltk
from config import get_config

kb_store = get_config()['KnowledgeBase']['knowledge_base_storage']
c_size = get_config()['KnowledgeBase']['chunk_size']
c_overlap = get_config()['KnowledgeBase']['chunk_overlap']


def get_stats():
    result = []

    for filename in os.listdir(kb_store):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_store, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            chunks = split_to_chunks(text, "md_headers")
            result.extend(chunks)

    df = pd.DataFrame({'strings': result})
    df['length'] = df['strings'].apply(len)

    interval_size = 50

    min_length = 0
    max_length = 500  # df['length'].max()

    bins = [i for i in range(int(min_length), int(max_length) + 1, int(interval_size))]
    if bins[-1] < max_length:
        bins.append(max_length)

    df['length_interval'] = pd.cut(df['length'], bins, right=False)
    interval_counts = df['length_interval'].value_counts().sort_index()
    interval_counts.to_csv('interval_counts.csv')


def get_articles_chunks():
    with open(kb_store + '/knowledge_base_new.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", text) if fragment.strip()]
    source_chunks = []
    for fragment in fragments:
        source_chunks.extend(markdown_splitter.split_text(fragment))

    with open(kb_store + '/knowledge_base_new2.txt', 'a', encoding='utf-8') as f:
        for chunk in source_chunks:

            if chunk.metadata:
                last_key = sorted(chunk.metadata.keys())[-1]
                header = chunk.metadata[last_key] + "\n"
                header_number = int(re.search(r'\d+', last_key).group())
                header_prefix = "#" * header_number

            f.write(header_prefix + " " + header + "\n" + header + str(chunk.page_content) + "\n\n")

    # count = 0
    #
    # with open(kb_store + '/chunks_statistics.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file, delimiter='|')
    #     writer.writerow(
    #         ['Filename', 'Article characters count', 'Chunks count', 'Chunks characters count', 'Lexical Diversity',
    #          'Noun Bigrams', 'Mean TF-IDF'])
    #
    #     for filename in os.listdir(kb_store):
    #         if filename.endswith(".txt"):
    #             file_path = os.path.join(kb_store, filename)
    #
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 text = f.read().strip()
    #
    #             chunks = split_to_chunks(text, "md_text")
    #             lengths = [len(chunk) for chunk in chunks]
    #             stats = check_chunks(chunks)
    #
    #             writer.writerow(
    #                 [filename, str(len(text)), str(len(chunks)), str(np.mean(lengths)), stats[0], stats[1], stats[2]])
    #
    #             count += 1
    #             percent = round((count / 84) * 100)
    #             print(f"\rОбработано текстов статей: {count} из {84} ({percent}%)", end='')


def split_to_chunks(text, split_type="md_headers"):
    result = []
    if split_type == "md_headers":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", text) if fragment.strip()]
        source_chunks = []
        for fragment in fragments:
            source_chunks.extend(markdown_splitter.split_text(fragment))
        for chunk in source_chunks:
            result.append(chunk)

    if split_type == "sentences":

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger_ru')

        sentences = sent_tokenize(text, language='russian')
        chunk_size = 7
        source_chunks = [sentences[x:x + chunk_size] for x in range(0, len(sentences), chunk_size)]

        for chunk in source_chunks:
            result.append(str(chunk))

    if split_type == "md_text":
        text_splitter = MarkdownTextSplitter(
            chunk_size=int(500), chunk_overlap=int(30)
        )
        fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", text) if fragment.strip()]
        for fragment in fragments:
            result.append("".join(text_splitter.split_text(fragment)))

    return result


def type_token_ratio(text):
    # Разбиваем текст на слова и считаем их частоту
    freq_dist = FreqDist(text.split())
    # Вычисляем Type-Token Ratio (TTR)
    return len(freq_dist) / len(text.split())


def lexical_diversity(chunks):
    return [type_token_ratio(chunk) for chunk in chunks]


def noun_bigrams(nlp, chunks):
    bigram_counts = []
    for chunk in chunks:
        doc = nlp(chunk)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        total_bigrams = len(set(ngrams(nouns, 2)))
        bigram_counts.append(total_bigrams)
    return bigram_counts


def mean_tfidf(chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return np.mean(tfidf_matrix.toarray(), axis=1).tolist()


def check_chunks(chunks):
    nlp = spacy.load("ru_core_news_sm")

    lex_div = lexical_diversity(chunks)
    bigram_counts = noun_bigrams(nlp, chunks)
    tfidf_scores = mean_tfidf(chunks)

    # print(f"Средняя лексическая разнообразность: {np.mean(lex_div)}")
    # print(f"Среднее количество биграмм именных сущностей: {np.mean(bigram_counts)}")
    # print(f"Средний TF-IDF: {np.mean(tfidf_scores)}")
    return [np.mean(lex_div), np.mean(bigram_counts), np.mean(tfidf_scores)]
