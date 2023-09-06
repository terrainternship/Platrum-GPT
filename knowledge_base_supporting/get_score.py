import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from llm_chat_bots import llm_init
from config import get_config

answers_storage = get_config()['GenAnswers']['answers_storage']

prompt_system_path = get_config()['GenScore']['prompt_system']
llm_name = get_config()['GenScore']['llm_name']
api_key = get_config()['GenScore']['api_key']
model_name = get_config()['GenScore']['model_name']
tpe_max_workers = get_config()['GenScore']['tpe_max_workers']
score_storage = get_config()['GenScore']['score_storage']
score_stats_storage = get_config()['GenScore']['score_stats_storage']

similarity_storage = get_config()['GenQuestionsSimilarity']['similarity_storage']


def process_score():
    # get_score()
    df = parse_score_data()
    clustering_and_save(df)


def clustering_and_save(df):
    df_from_data_list = pd.DataFrame(df)
    features = df_from_data_list[['accuracy', 'completeness', 'clarity', 'contextual_relevance']]
    kmeans_from_data_list = kmeans = KMeans(n_clusters=10, n_init=10, random_state=0).fit(features)
    df_from_data_list['cluster'] = kmeans_from_data_list.labels_
    df_from_data_list.to_csv(score_stats_storage, index=False, sep='|', encoding='utf-8')


def parse_score_data():
    with open(score_storage, 'r', encoding='utf-8') as file:
        score_store = json.load(file)

    data_clustering= []
    for frame in score_store['frames']:

        article_id = frame['article_id']
        chunk_id = frame['chunk_id']
        question = clean_string(frame['question'])
        answer = clean_string(frame['answer'])

        score_str = fix_invalid_json(frame['score'].replace('\\\\', '\\').replace('\\"', '"'))
        score_items = json.loads(score_str).items()

        values = [v for k, v in score_items]
        accuracy = values[0] if len(values) > 0 else None
        completeness = values[1] if len(values) > 1 else None
        clarity = values[2] if len(values) > 2 else None
        contextual_relevance = values[3] if len(values) > 3 else None

        data_dict_for_clustering = {
            'article_id': article_id,
            'chunk_id': chunk_id,
            'question': question,
            'answer': answer,
            'accuracy': accuracy,
            'completeness': completeness,
            'clarity': clarity,
            'contextual_relevance': contextual_relevance
        }
        data_clustering.append(data_dict_for_clustering)
    return data_clustering


def clean_string(s):
    return s.replace("\t", " ").replace("\n", " ").replace("|", " ").replace(";", " ")


def fix_invalid_json(json_str):
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r'}[^}]*$', '}', json_str)
    return json_str


def get_score():
    with open(similarity_storage, 'r', encoding='utf-8') as file:
        similarity_store = json.load(file)

    similar_questions_list = [item["question"] for item in similarity_store["questions"]]

    with open(answers_storage, 'r', encoding='utf-8') as file:
        answers_store = json.load(file)

    with open(prompt_system_path, 'r') as f:
        system = f.read().strip()

    config = {
        'name': llm_name,
        'api_key': api_key,
        'model': model_name,
        'system_message': system
    }

    chat_bot = llm_init(config)

    data = {
        "frames": []
    }

    count = 0
    future_to_m = {}
    with ThreadPoolExecutor(max_workers=int(tpe_max_workers)) as executor:
        futures = []

        for frame in answers_store['frames']:
            article_id = frame['article_id']
            chunk_id = frame['chunk_id']
            question = frame['question']
            answer = frame['answer']

            if question in similar_questions_list:
                continue

            count += 1
            m = {
                "article_id": article_id,
                "chunk_id": chunk_id,
                "question": question,
                "answer": answer,
                "score": ""
            }

            future = executor.submit(
                lambda m=m: chat_bot.chat(f'"question": "{m["question"]}",\n"answer": "{m["answer"]}"'))
            futures.append(future)
            future_to_m[future] = m

        for future in as_completed(futures):
            m = future_to_m[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred during future execution: {e}")
                continue

            m['score'] = result
            data['frames'].append(m)

            percent = round(count / (len(answers_store['frames']) * 100))
            print(f"\rОбработка ответов: оценено {count} из {len(answers_store['frames'])} ({percent}%)", end='')

    if data:
        with open(score_storage, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
