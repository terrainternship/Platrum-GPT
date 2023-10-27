import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, FAISS

from llm_chat_bots import llm_init

from config import get_config

kb_store = get_config()['KnowledgeBase']['knowledge_base_storage']
prompt_system_path = get_config()['GenFix']['prompt_system']
llm_name = get_config()['GenFix']['llm_name']
api_key = get_config()['GenFix']['api_key']
model_name = get_config()['GenFix']['model_name']
tpe_max_workers = get_config()['GenFix']['tpe_max_workers']
fix_store = get_config()['GenFix']['fix_storage']


def get_clear_text():
    # algo_fixing()
    # get_smart_headers()
    # create_vector_db()
    # algo_fixing_video()
    # create_vector_db_video()
    # work_vector_db()
    # compare_vectors_db()
    # create_vector_db_questions()
    work_vector_db_questions()

def compare_vectors_db():
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    # Compare vector storage for text self
    # dbt = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_text')
    # print("Vestor store for main text succesfully loaded from " + fix_store + '/vectorstore')
    #
    # with open(fix_store + '/knowledge_base_fix.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # with open(fix_store + '/compare_text_self.txt', 'a', encoding='utf-8') as f:
    #     for item in data:
    #         question = item['chunk_smart_header'] + '\n' + item['chunk']
    #         docs = dbt.similarity_search_with_score(question, 2)
    #
    #         for doc, score in docs:
    #             if question != doc.page_content:
    #                 f.write(str(score) + '\n')
    #                 break

    # Compare vector storage for video self
    # dbv = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_video')
    # print("Vestor store for video succesfully loaded from " + fix_store + '/vectorstore')
    #
    # with open(fix_store + '/video/knowledge_base_video_fix.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # with open(fix_store + '/compare_video_self.txt', 'a', encoding='utf-8') as f:
    #     for item in data:
    #         question = item['chunk_header'] + item['chunk']
    #         docs = dbv.similarity_search_with_score(question, 2)
    #
    #         for doc, score in docs:
    #             if question != doc.page_content:
    #                 f.write(str(score) + '\n')
    #                 break

    # Compare vector storage for text in video self
    # dbv = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_video')
    # print("Vestor store for video succesfully loaded from " + fix_store + '/vectorstore')
    #
    # with open(fix_store + '/knowledge_base_fix.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # with open(fix_store + '/compare_text_video_self.txt', 'a', encoding='utf-8') as f:
    #     for item in data:
    #         question = item['chunk_smart_header'] + '\n' + item['chunk']
    #         docs = dbv.similarity_search_with_score(question, 2)
    #
    #         for doc, score in docs:
    #             if question != doc.page_content:
    #                 f.write(str(score) + '\n')
    #                 break

    # Compare vector storage for video in text self
    dbt = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_text')
    print("Vestor store for main text succesfully loaded from " + fix_store + '/vectorstore')

    with open(fix_store + '/video/knowledge_base_video_fix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(fix_store + '/compare_video_text_self.txt', 'a', encoding='utf-8') as f:
        for item in data:
            question = item['chunk_header'] + item['chunk']
            docs = dbt.similarity_search_with_score(question, 2)

            for doc, score in docs:
                if question != doc.page_content:
                    f.write(str(score) + '\n')
                    break


def work_vector_db():
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    dbt = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_text')
    print("Vestor store for main text succesfully loaded from " + fix_store + '/vectorstore')

    dbv = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_video')
    print("Vestor store for video succesfully loaded from " + fix_store + '/vectorstore')

    with open(prompt_system_path, 'r') as f:
        system = f.read().strip()

    config = {
        'name': llm_name,
        'api_key': api_key,
        'model': model_name,
        'prompt': system
    }

    chat_bot = llm_init(config)

    with open('./knowledge_base_supporting/questions/target.txt', 'r') as file:
        questions = file.readlines()

    questions = [question.strip() for question in questions]
    qcount = len(questions)
    count = 0
    with open(fix_store + '/target_stats.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerow(["question", "chunk_from_text", "chunk_from_video", "answer"])

        for question in questions:
            docs = dbt.similarity_search_with_score(question, 1)
            docv = dbv.similarity_search_with_score(question, 1)

            chunk_from_text = "score: " + str(docs[0][1]) + ', ' + \
                              docs[0][0].metadata['chunk']

            chunk_from_video = "score: " + str(docv[0][1]) + ', ' + \
                               docv[0][0].metadata['chunk']

            answer = ""
            # answer = chat_bot.completion("question: " + question + "\n" + \
            #                              "material#1: " + chunk_from_text + "\n" + \
            #                              "material#2: " + chunk_from_video
            #                              )
            csvwriter.writerow(
                [question.replace("\n", "").replace("\t", ""),
                 chunk_from_text.replace("\n", "").replace("\t", ""),
                 chunk_from_video.replace("\n", "").replace("\t", ""),
                 answer.replace("\n", "").replace("\t", "")])

            count += 1
            percent = round((count / qcount) * 100)
            print(f"\rОбработка вопросов: выполнено {count} из {qcount} ({percent}%)", end='')

    # while True:
    #     question = input("Введите ваш вопрос (или 'q'/'quit' для выхода): ").strip()
    #
    #     if question.lower() in ['q', 'quit']:
    #         break
    #
    #     print("-------------------------TEXT----------------------------")
    #     docs = dbt.similarity_search_with_score(question, 4)
    #     for doc, score in docs:
    #         print("score: " + str(score))
    #         print("article_id: " + doc.metadata['article_id'])
    #         print("chunk_size: " + str(doc.metadata['chunk_size']))
    #         print("chunk_header_raw: " + str(doc.metadata['chunk_header_raw']))
    #         print("chunk_smart_header: " + doc.metadata['chunk_smart_header'])
    #         print("chunk: " + doc.page_content)
    #         print("---------------------------------------------------------")
    #
    #     print("-------------------------VIDEO---------------------------")
    #     docs = dbv.similarity_search_with_score(question, 4)
    #     for doc, score in docs:
    #         print("score: " + str(score))
    #         print("article_id: " + doc.metadata['article_id'])
    #         print("chunk_size: " + str(doc.metadata['chunk_size']))
    #         print("chunk_header_raw: " + str(doc.metadata['chunk_header_raw']))
    #         print("chunk: " + doc.page_content)
    #         print("---------------------------------------------------------")


def create_vector_db():
    with open(fix_store + '/knowledge_base_fix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=item['chunk_smart_header'] + '\n' + item['chunk'],
            metadata={
                'chunk_size': item['chunk_size'],
                'article_id': item['article_id'],
                'chunk_smart_header': item['chunk_smart_header'],
                'chunk_header_raw': item['chunk_header_raw'],
                'chunk': item['chunk']
            }
        ) for item in data
    ]

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    print("Vestor store succesfully created")
    db.save_local(fix_store + '/vectorstore', 'index_text')
    print("Vestor store succesfully saved in " + fix_store + '/vectorstore')


def create_vector_db_video():
    with open(fix_store + '/video/knowledge_base_video_fix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=item['chunk_header'] + item['chunk'],
            metadata={
                'chunk_size': item['chunk_size'],
                'article_id': item['article_id'],
                'chunk_header_raw': item['chunk_header_raw'],
                'chunk': item['chunk']
            }
        ) for item in data
    ]

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    print("Vestor store succesfully created")
    db.save_local(fix_store + '/vectorstore', 'index_video')
    print("Vestor store succesfully saved in " + fix_store + '/vectorstore')


def create_vector_db_questions():
    with open(fix_store + '/questions/data_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        chunk_id = item['chunk_id']
        questions = item['questions']

        for question in questions:
            doc = Document(
                page_content=question,
                metadata={
                    'chunk_id': chunk_id,
                }
            )
            documents.append(doc)

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    print("Vestor store succesfully created")
    db.save_local(fix_store + '/vectorstore', 'index_questions')
    print("Vestor store succesfully saved in " + fix_store + '/vectorstore')


def work_vector_db_questions():
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    dbq = FAISS.load_local(fix_store + '/vectorstore', OpenAIEmbeddings(), 'index_questions')
    print("Vestor store for questions succesfully loaded from " + fix_store + '/vectorstore')

    while True:
        question = input("Введите ваш вопрос (или 'q'/'quit' для выхода): ").strip()

        if question.lower() in ['q', 'quit']:
            break

        print("-------------------------QUESTIONS----------------------------")
        docs = dbq.similarity_search_with_score(question, 4)
        for doc, score in docs:
            print("score: " + str(score))
            print("chunk_id: " + doc.metadata['chunk_id'])
            print("chunk: " + doc.page_content)
            print("---------------------------------------------------------")

def get_smart_headers():
    with open(prompt_system_path, 'r') as f:
        system = f.read().strip()

    config = {
        'name': llm_name,
        'api_key': api_key,
        'model': model_name,
        'system_message': system
    }

    chat_bot = llm_init(config)

    count = 0
    data = []
    for filename in os.listdir(fix_store + '/old'):

        if filename.endswith(".json"):
            file_path = os.path.join(fix_store + '/old', filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            count += 1

            for chunk in chunks:
                article_id = chunk['article_id']
                chunk_header_raw = chunk['chunk_header_raw']
                chunk_text = chunk['chunk']

                result = chat_bot.chat(chunk_text)

                m = {
                    "article_id": article_id,
                    "chunk_header_raw": chunk_header_raw,
                    "chunk_smart_header": result.replace('"', ''),
                    "chunk_size": len(chunk_text),
                    "chunk": chunk_text
                }

                data.append(m)
            percent = round((count / 84) * 100)
            print(f"\rОбработка вопросов: выполнено {count} из 84 ({percent}%)", end='')

    if data:
        with open(fix_store + '/knowledge_base_fix.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # chunk_text = "Число проектов, которое вы можете создать - неограниченно. Вы также можете пригласить в разные проекты одних и тех же пользователей. Это может быть востребовано в случае, если у вас несколько направлений в бизнесе и вы хотите разделить их в Platrum.\nКаждый проект работает обособлено от других. Это означает, что для каждого проекта вам нужно будет приобрести свою лицензию (в случае необходимости), добавить сотрудников, наполнить данными. Данные в разных проектах одного владельца никак не синхронизируются между собой.\nПри этом пользователь может быть как владельцем одного проекта, так и рядовым сотрудником в другом. Для создания нового проекта нужно нажать по иконке профиля слева вверху - Проекты - Добавить проект.\nhttps://s3.amazonaws.com/cdn.freshdesk.com/data/helpdesk/attachments/production/73003083808/original/yy_sV7FlSeUVQMf4hdxdzOvl-rK-p86hmw.png?1645106690\nЧтобы переключаться между несколькими проектами, необходимо нажать на ваш аватар и далее на название текущего проекта - откроется список всех ваших проектов.\nhttps://s3.amazonaws.com/cdn.freshdesk.com/data/helpdesk/attachments/production/73003083817/original/ksJdh4hFBshoOCCDoV8LMYaEZIb3W0xsWw.png?1645106708"
    # print(system)
    # print("----------------")
    # print(chat_bot.chat(chunk_text))

    # count = 0
    # future_to_m = {}
    # data = []
    # with ThreadPoolExecutor(max_workers=int(tpe_max_workers)) as executor:
    #     futures = []
    #
    #     for filename in os.listdir(fix_store + '/old'):
    #         if filename.endswith(".json"):
    #             file_path = os.path.join(fix_store + '/old', filename)
    #
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 chunks = json.load(f)
    #             count += 1
    #             for chunk in chunks:
    #                 article_id = chunk['article_id']
    #                 chunk_header_raw = chunk['chunk_header_raw']
    #                 chunk_text = chunk['chunk']
    #
    #                 m = {
    #                     "article_id": article_id,
    #                     "chunk_header_raw": chunk_header_raw,
    #                     "chunk_smart_header": "",
    #                     "chunk_size": len(chunk_text),
    #                     "chunk": chunk_text
    #                 }
    #
    #                 future = executor.submit(lambda m=m: chat_bot.chat(chunk_text))
    #                 futures.append(future)
    #                 future_to_m[future] = m
    #
    #     for future in as_completed(futures):
    #         m = future_to_m[future]
    #
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f"An error occurred during future execution: {e}")
    #             continue
    #
    #         m['chunk_smart_header'] = result
    #         data.append(m)
    #
    #         percent = round((count / 84) * 100)
    #         print(f"\rОбработка вопросов: выполнено {count} из 84 ({percent}%)", end='')
    #
    # if data:
    #     with open(fix_store + '/knowledge_base_fix.json', 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=4)


def algo_fixing():
    for filename in os.listdir(kb_store):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_store, filename)

            identifier = re.search(r'_(\d+)\.', filename)
            identifier = identifier.group(1) if identifier else None

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            processed_lines = []
            remove_flag = False
            pattern = re.compile(r'\(#[^\)]*\)')

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                line = pattern.sub('', line)

                if line.startswith('СОДЕРЖАНИЕ'):
                    remove_flag = True
                    continue
                if remove_flag and line.startswith('-'):
                    continue

                processed_lines.append(line)

            text = '\n'.join(processed_lines)

            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5")
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", text) if fragment.strip()]
            source_chunks = []
            for fragment in fragments:
                source_chunks.extend(markdown_splitter.split_text(fragment))
            chunks = []
            for chunk in source_chunks:
                header = ""
                if chunk.metadata:
                    last_key = sorted(chunk.metadata.keys())[-1]
                    header = chunk.metadata[last_key] + "\n"

                b = {
                    "article_id": identifier,
                    "chunk_header_raw": chunk.metadata,
                    "chunk_header": header,
                    "chunk_size": len(chunk.page_content),
                    "chunk": chunk.page_content
                }

                chunks.append(b)

            if chunks:
                with open(fix_store + '/old/' + filename.replace(".txt", ".json"), 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)


def algo_fixing_video():
    mappings = {
        'https://www.youtube.com/watch?v=n4hzqgVVGDQ&t=0.0s': '73000524812',
        'n4hzqgVVGDQ': '73000524812',
        'https://www.youtube.com/watch?v=oplGA717U9I&t=0.0s': '73000524812',
        'oplGA717U9I': '73000524812',
        'https://www.youtube.com/watch?v=7RXYl8Ja5Hs&t=0.0s': '73000525555',
        '7RXYl8Ja5Hs': '73000525555',
        'https://www.youtube.com/watch?v=2SZMYKNSBQk&t=0.0s': '73000525532',
        '2SZMYKNSBQk': '73000525532',
        'https://www.youtube.com/watch?v=2SZMYKNSBQk&t=258.0s': '73000525535',
        'https://www.youtube.com/watch?v=2SZMYKNSBQk&t=558.0s': '73000525540',
        'https://www.youtube.com/watch?v=2SZMYKNSBQk&t=1430.0s': '73000544921',
        'https://www.youtube.com/watch?v=eDWWqz2Hlkc&t=0.0s': '73000525545',
        'eDWWqz2Hlkc': '73000525545',
        'https://www.youtube.com/watch?v=Ow3K61Bi1Xk&t=0.0s': '73000525543',
        'Ow3K61Bi1Xk': '73000525543',
        'https://www.youtube.com/watch?v=BmqF2xw-q64&t=0.0s': '73000525771',
        'BmqF2xw-q64': '73000525771',
        'https://www.youtube.com/watch?v=pVC-GojdDwo&t=0.0s': '73000373435',
        'pVC-GojdDwo': '73000373435',
        'https://www.youtube.com/watch?v=pVC-GojdDwo&t=41.0s': '73000525805',
        'https://www.youtube.com/watch?v=pVC-GojdDwo&t=176.0s': '73000525802',
        'https://www.youtube.com/watch?v=EHVMUtO5Kpw&t=0.0s': '73000525814',
        'EHVMUtO5Kpw': '73000525814',
        'https://www.youtube.com/watch?v=60KUgdiURm8&t=0.0s': '73000525814',
        '60KUgdiURm8': '73000525814',
        'https://www.youtube.com/watch?v=A5XJcH9EYZs&t=0.0s': '73000525814',
        'A5XJcH9EYZs': '73000525814',
        'https://www.youtube.com/watch?v=v9ob-3hPq-Y&t=0.0s': '73000588898',
        'v9ob-3hPq-Y': '73000588898'
    }

    with open(fix_store + '/knowledge_base_fix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for filename in os.listdir(fix_store + '/video'):
        if filename.endswith(".txt"):
            file_path = os.path.join(fix_store + '/video', filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5")
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", "".join(lines)) if
                         fragment.strip()]
            source_chunks = []
            for fragment in fragments:
                source_chunks.extend(markdown_splitter.split_text(fragment))
            chunks = []
            for chunk in source_chunks:

                pattern = re.compile(r'\[.*?\]\(https://www\.youtube\.com/watch\?v=[a-zA-Z0-9_-]+\)', re.IGNORECASE)
                if pattern.fullmatch(chunk.page_content):
                    continue

                header = ""
                if chunk.metadata:
                    last_key = sorted(chunk.metadata.keys())[-1]
                    header = chunk.metadata[last_key] + "\n"
                article_id = ""
                youtube_link_pattern = re.compile(r'\(https://www\.youtube\.com/watch\?v=([a-zA-Z0-9_-]+)&t=[0-9.]+s\)')
                match = youtube_link_pattern.search(chunk.page_content)
                if match:
                    youtube_id = match.group(1)
                    youtube_full_link = match.group(0)[1:-1]
                    for frame in data:
                        if 'chunk' in frame and youtube_id in frame['chunk']:
                            article_id = frame.get('article_id')
                    if not article_id:
                        article_id = mappings.get(youtube_full_link)
                        if not article_id:
                            article_id = mappings.get(youtube_id)
                            if not article_id:
                                print("Error, not found chunk in KB for: " + youtube_full_link + " # " + youtube_id)
                else:
                    print("Error, not found link in KB_video: " + chunk.page_content)

                text = re.sub(r'\[.*?\]\(https://www\.youtube\.com/watch\?v=[a-zA-Z0-9_-]+&t=[0-9.]+s\)\n?', '',
                              chunk.page_content)
                text = text.strip()

                b = {
                    "article_id": article_id,
                    "chunk_header_raw": chunk.metadata,
                    "chunk_header": header,
                    "chunk_size": len(text),
                    "chunk": text
                }

                chunks.append(b)

            if chunks:
                with open(fix_store + '/video/' + filename.replace(".txt", ".json"), 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)
