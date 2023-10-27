import csv
import json
import os
import uuid

import openai
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, FAISS
import pandas as pd
import numpy as np

from collections import defaultdict
from config import get_config

api_key = get_config()['GenFix']['api_key']
fix_store = get_config()['GenFix']['fix_storage']

similarity_search_chunks_count = 4
similarity_search_questions_count = 4

os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

db = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_chunks_kb')
print("Vestor store for chunks succesfully loaded from " + fix_store + '/vectorstore/index_chunks_kb')

with open(fix_store + '/knowledge_base.json', 'r', encoding='utf-8') as f:
    db_plain = json.load(f)

db_plain_dict = {}
for item in db_plain:
    chunk_id = item['chunk_id']
    chunk = item['chunk']
    db_plain_dict[chunk_id] = chunk
print("Dictionary for chunks succesfully loaded from " + fix_store + '/knowledge_base.json')

dbq = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_questions_kb')
print("Vestor store for chunks succesfully loaded from " + fix_store + '/vectorstore/index_questions_kb')

with open(fix_store + '/questions/data_1.json', 'r', encoding='utf-8') as f:
    dbq_plain = json.load(f)

dbq_plain_dict = {}
for item in dbq_plain:
    chunk_id = item['chunk_id']
    question_id = item['question_id']
    if chunk_id in dbq_plain_dict:
        dbq_plain_dict[chunk_id].append(question_id)
    else:
        dbq_plain_dict[chunk_id] = [question_id]
print("Dictionary for chunks succesfully loaded from " + fix_store + '/knowledge_base.json')


def get_answer(user_question, verbose=0):
    chunks_docs = db.similarity_search_with_score(user_question, similarity_search_chunks_count)

    if verbose:
        print("-------------------------CHUNKS----------------------------")
        for doc, score in chunks_docs:
            print("score: " + str(score))
            print("chunk_id: " + doc.metadata['chunk_id'])
            print("chunk_smart_header: " + doc.metadata['chunk_smart_header'])
            print("chunk: " + doc.page_content)
            print("---------------------------------------------------------")

    final_chunks_scores = defaultdict(float)
    final_questions_scores = defaultdict(float)

    for doc, chunk_score in chunks_docs:
        chunk_id = doc.metadata['chunk_id']
        final_chunks_scores[chunk_id] = 1 / chunk_score
        if verbose:
            print(f"chunk_id: {chunk_id}, score {chunk_score}")
        questions = dbq_plain_dict.get(chunk_id)
        avg_score = 0
        count = 0
        for question_id in questions:
            questions_docs = dbq.similarity_search_with_score(
                user_question,
                filter={"question_id": question_id}
            )
            if verbose:
                print(f"chunk_id: {chunk_id}, question_id {question_id}")
            for _, score in questions_docs:
                final_questions_scores[question_id] += 1 / score
                avg_score += score
                count += 1
                if verbose:
                    print(f"chunk_id: {chunk_id}, question_id {question_id}, score {score}")
        if avg_score:
            if verbose:
                print(f"chunk_id: {chunk_id}, avg questions score {(avg_score / count)}")
            final_chunks_scores[chunk_id] += 1 / (avg_score / count)

    bot_questions_docs = dbq.similarity_search_with_score(user_question, similarity_search_questions_count)

    if verbose:
        print("-------------------------QUESTIONS----------------------------")
        for doc, score in bot_questions_docs:
            print("score: " + str(score))
            print("chunk_id: " + doc.metadata['chunk_id'])
            print("question_id: " + doc.metadata['question_id'])
            print("question: " + doc.page_content)
            print("---------------------------------------------------------")

    score_accumulator = defaultdict(lambda: {'sum': 0, 'count': 0})
    for doc, score in bot_questions_docs:
        final_questions_scores[doc.metadata['question_id']] += 1 / score
        chunk_id = doc.metadata['chunk_id']
        accumulator = score_accumulator[chunk_id]
        accumulator['sum'] += score
        accumulator['count'] += 1

    average_scores = {chunk_id: data['sum'] / data['count'] for chunk_id, data in score_accumulator.items()}
    for chunk_id, avg_score in average_scores.items():
        final_chunks_scores[chunk_id] = 1 / avg_score
        bot_questions_chunks_docs = db.similarity_search_with_score(
            user_question,
            filter={"chunk_id": chunk_id}
        )
        for doc, score in bot_questions_chunks_docs:
            final_chunks_scores[chunk_id] += 1 / score

    return final_chunks_scores, final_questions_scores


def get_clear_answer():
    # create_chunks_vector_db()
    # reformat_db_questions()
    # create_vector_db_questions()
    # work_auto()
    classify_answers()


def work_auto():
    with open('./knowledge_base_supporting/questions/target.txt', 'r') as file:
        questions = file.readlines()

    questions = [question.strip() for question in questions]
    qcount = len(questions)
    count = 0

    with open(fix_store + '/target_vectors.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerow(["question", "first_chunk", "first_question", "chunks", "questions"])

        for question in questions:
            chunks, questions = get_answer(question)
            chunks_first_key, _ = next(iter(chunks.items()))
            questions_first_key, _ = next(iter(questions.items()))
            chunks_str = ", ".join(map(str, sorted(chunks.values(), reverse=True)))
            questions_str = ", ".join(map(str, sorted(questions.values(), reverse=True)))
            # print(f"{question}: {chunks_first_key}, {questions_first_key}, {chunks_str}, {questions_str}")

            csvwriter.writerow(
                [question.replace("\n", "").replace("\t", ""),
                 chunks_first_key.replace("\n", "").replace("\t", ""),
                 questions_first_key.replace("\n", "").replace("\t", ""),
                 chunks_str.replace("\n", "").replace("\t", ""),
                 questions_str.replace("\n", "").replace("\t", "")])

            count += 1
            percent = round((count / qcount) * 100)
            print(f"\rОбработка вопросов: выполнено {count} из {qcount} ({percent}%)", end='')


def work_manual():
    while True:
        question = input("Введите ваш вопрос (или 'q'/'quit' для выхода): ").strip()

        if question.lower() in ['q', 'quit']:
            break

        chunks, questions = get_answer(question)

        scores = list(questions.values())
        percentile_90 = np.percentile(scores, 75)
        print("Статистика questions:")
        for question_id, total_score in sorted(questions.items(), key=lambda item: item[1], reverse=True):
            if total_score >= percentile_90:
                print(f"question_id: {question_id}, total score: {total_score}")

        scores = list(chunks.values())
        percentile_90 = np.percentile(scores, 75)
        print("Статистика chunks:")
        for chunk_id, total_score in sorted(chunks.items(), key=lambda item: item[1], reverse=True):
            if total_score >= percentile_90:
                print(f"chunk_id: {chunk_id}, total score: {total_score}")


def reformat_db_questions():
    with open(fix_store + '/questions/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_questions = []
    for chunk in data:
        chunk_id = chunk['chunk_id']
        questions = chunk['questions']

        for question in questions:
            formatted_question = {
                "chunk_id": chunk_id,
                "question_id": str(uuid.uuid4()),
                "question": question
            }
            formatted_questions.append(formatted_question)

    with open(fix_store + '/questions/data_1.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_questions, f, ensure_ascii=False, indent=4)


def create_chunks_vector_db():
    with open(fix_store + '/knowledge_base.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=item['chunk_smart_header'] + '\n' + item['chunk'],
            metadata={
                'article_id': item['article_id'],
                'chunk_header_raw': item['chunk_header_raw'],
                'chunk_smart_header': item['chunk_smart_header'],
                'chunk_size': item['chunk_size'],
                'chunk': item['chunk'],
                'chunk_id': item['chunk_id'],
            }
        ) for item in data
    ]

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    print("Vestor store succesfully created")
    db.save_local(fix_store + '/vectorstore', 'index_chunks_kb')
    print("Vestor store succesfully saved in " + fix_store + '/vectorstore/index_chunks_kb')


def create_vector_db_questions():
    with open(fix_store + '/questions/data_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        chunk_id = item['chunk_id']
        question_id = item['question_id']
        question = item['question']

        doc = Document(
            page_content=question,
            metadata={
                'chunk_id': chunk_id,
                'question_id': question_id,
            }
        )
        documents.append(doc)

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    print("Vestor store succesfully created")
    db.save_local(fix_store + '/vectorstore', 'index_questions_kb')
    print("Vestor store succesfully saved in " + fix_store + '/vectorstore/index_questions_kb')


def parse_numerical_list(numerical_str):
    return [float(num) for num in numerical_str.split(',')]


def classify_answers():
    data = pd.read_csv(fix_store + '/target_vectors.csv', delimiter='|')
    data['chunks'] = data['chunks'].apply(lambda x: [float(i) for i in x.split(',')])
    data['questions'] = data['questions'].apply(lambda x: [float(i) for i in x.split(',')])
    data['decision'] = data.apply(analyze_and_decide, axis=1)
    data.to_csv(fix_store + '/target_vectors_classify.csv', sep='|', index=False)


def analyze_and_decide(row):
    threshold_good_chunk = 7.47
    threshold_good_question = 6.57

    threshold_low_chunk = 3.56
    threshold_low_question = 3.68

    max_chunk_value = max(row['chunks']) if row['chunks'] else 0
    max_question_value = max(row['questions']) if row['questions'] else 0

    decision = {"show_chunks": 0, "show_questions": 0, "category": "Неопределенная категория"}

    if max_chunk_value < threshold_low_chunk and max_question_value < threshold_low_question:
        decision["category"] = "Вопрос не относится к деятельности сервиса"
    else:

        if max_chunk_value < threshold_good_chunk:
            deviation_threshold_chunk = 0.05
        else:
            deviation_threshold_chunk = 0.15

        if max_question_value < threshold_good_question:
            deviation_threshold_question = 0.05
        else:
            deviation_threshold_question = 0.15

        good_chunks = [index for index, value in enumerate(row['chunks']) if
                       value >= threshold_good_chunk and
                       (max_chunk_value - value) / max_chunk_value <= deviation_threshold_chunk]

        good_questions = [index for index, value in enumerate(row['questions']) if
                          value >= threshold_good_question and
                          (max_question_value - value) / max_question_value <= deviation_threshold_question]

    if not good_chunks and not good_questions:
        decision["category"] = "Нет данных в БЗ, попробуйте переформулировать вопрос"
    elif good_chunks and not good_questions:
        decision["category"] = "Есть данные в БЗ"
        decision["show_chunks"] = len(good_chunks)
    elif good_questions and not good_chunks:
        decision["category"] = "Нет данных в БЗ, попробуйте вопрос из списка предложенных"
        decision["show_questions"] = len(good_questions)
    elif good_chunks and good_questions:
        decision["category"] = "Есть данные в БЗ, если не получили ответ, попробуйте вопрос из списка предложенных"
        decision["show_chunks"] = len(good_chunks)
        decision["show_questions"] = len(good_questions)

    return decision
