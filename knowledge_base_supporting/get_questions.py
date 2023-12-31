import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_chat_bots import llm_init
from config import get_config

kb_store = get_config()['KnowledgeBase']['knowledge_base_storage']

prompt_system_path = get_config()['GenQuestions']['prompt_system']
llm_name = get_config()['GenQuestions']['llm_name']
api_key = get_config()['GenQuestions']['api_key']
model_name = get_config()['GenQuestions']['model_name']
tpe_max_workers = get_config()['GenQuestions']['tpe_max_workers']
questions_storage = get_config()['GenQuestions']['questions_storage']


def questions_from_llm():

    with open(kb_store + '/knowledge_base.json', 'r', encoding='utf-8') as file:
        knowledge_base = json.load(file)

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
        'total_chunks': 0,
        'total_questions': 0,
        "chunks": []
    }

    future_to_m = {}

    with ThreadPoolExecutor(max_workers=int(tpe_max_workers)) as executor:
        futures = []

        for article in knowledge_base:
            for chunk in article['content']['chunks']:
                m = {
                    "id": chunk['id'],
                    "article_id": article['id'],
                    "questions": ""
                }
                future = executor.submit(lambda m=m: chat_bot.chat(str(chunk['en'])))
                futures.append(future)
                future_to_m[future] = m

        for future in as_completed(futures):
            m = future_to_m[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred during future execution: {e}")
                continue

            data['total_chunks'] += 1

            lines = result.split("\n")
            questions = []
            for line in lines:
                line = process_and_filter_line(line.strip())
                if not line:
                    continue
                if re.match(r'^[a-zA-Z]', line):
                    continue
                if '?' not in line:
                    continue
                questions.append(line)

            data['total_questions'] += len(questions)
            m['questions'] = questions
            data['chunks'].append(m)

            print(f"\rОбработка чанка: {data['total_chunks']}", end='')

    if data['total_chunks'] > 0:
        with open(questions_storage, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def process_and_filter_line(line: str) -> str:
    processed_line = re.sub(r'^\d+\.\s*', '', line)
    processed_line = re.sub(r'^-\s*', '', processed_line)
    return processed_line


def process_questions():
    questions_from_llm()
