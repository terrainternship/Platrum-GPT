import json
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge_base_updating.get_articles_url import get_html
from knowledge_base_updating.get_chunks import split_to_chunks_md_text
from llm_chat_bots import llm_init
from config import get_config

kb_cache = get_config()['KnowledgeBase']['cache']
kb_store = get_config()['KnowledgeBase']['knowledge_base_storage']
kb_articles_sep = get_config()['KnowledgeBase']['knowledge_base_save_translated_articles']
chunk_size = get_config()['KnowledgeBase']['chunk_size']
chunk_overlap = get_config()['KnowledgeBase']['chunk_overlap']
prompt_system_path = get_config()['KnowledgeBase']['prompt_system']
llm_name = get_config()['KnowledgeBase']['llm_name']
api_key = get_config()['KnowledgeBase']['api_key']
model_name = get_config()['KnowledgeBase']['model_name']
tpe_max_workers = get_config()['KnowledgeBase']['tpe_max_workers']


def get_articles_text():
    with open(kb_cache, 'r', encoding='utf-8') as file:
        data = json.load(file)

    total_articles = 0
    for section_name, section in data['sections'].items():
        for _ in section['articles']:
            total_articles += 1

    count = 0
    articles = []
    for section_name, section_content in data['sections'].items():
        for article_name, article_content in section_content['articles'].items():

            html = get_html(article_content['url'])

            article_id = re.search(r'/articles/(\d+)-', article_content['url'])
            if article_id:
                article_id = article_id.group(1)
            else:
                article_id = None

            soup = BeautifulSoup(html, 'html.parser')
            article = soup.find('article', id='article-body')
            if not article:
                continue

            kb = process_article(article)
            kb.insert(0, "\n" + "# " + article_name + "\n\n")

            if kb_articles_sep:
                kb_article = {
                    "id": article_id,
                    "title": article_name,
                    "content": {
                        "chunks": []
                    }
                }
                articles.append(split_and_translate(kb, kb_article))

            else:
                save_text(kb, kb_store + '/knowledge_base.txt')

            count += 1
            percent = round((count / total_articles) * 100)
            print(f"\rСохранение текста статей: выполнено {count} из {total_articles} ({percent}%)", end='')

    if kb_articles_sep and articles:
        with open(kb_store + '/knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)


def split_and_translate(kb: list[str], kb_article):

    kb = "\n".join(kb)
    splits = split_to_chunks_md_text(kb, chunk_size, chunk_overlap)

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
    future_to_m = {}
    with ThreadPoolExecutor(max_workers=int(tpe_max_workers)) as executor:
        futures = []

        for split in splits:
            count += 1
            chunk = {
                "id": count,
                "ru": str(split.page_content),
                "en": ""
            }
            future = executor.submit(lambda chunk=chunk: chat_bot.chat(str(chunk['ru'])))
            futures.append(future)
            future_to_m[future] = chunk

        for future in as_completed(futures):
            chunk = future_to_m[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred during future execution: {e}")
                continue
            chunk['en'] = result
            kb_article["content"]["chunks"].append(chunk)

    return kb_article


def save_text(text, file):
    with open(file, 'a', encoding='utf-8') as f:
        for item in text:
            f.write(str(item) + "\n")


def process_article(element, level=0):
    result = []

    if isinstance(element, str):
        stripped_str = element.strip()
        if stripped_str:
            result.append(stripped_str)
    else:
        if element.name and element.name.startswith('h') and element.name[1].isdigit():
            level = int(element.name[1]) + 1
            result.append("\n" + "#" * level + " " + element.get_text().strip())

        elif element.name == 'img':
            img_src = element.get('src', '')
            if img_src:
                result.append(img_src)

        elif element.name == 'iframe':
            iframe_src = element.get('src', '')
            if iframe_src:
                result.append(iframe_src)

        elif element.name == 'a':
            anchor_text = element.get_text().strip()
            href = element.get('href', '')
            result.append(f"{anchor_text} ({href})")

        elif element.name == 'ul' or element.name == 'ol':
            for li in element.find_all('li', recursive=False):
                result.extend(process_article(li, level + 1))

        elif element.name == 'li':
            main_text = element.get_text().strip()
            for nested_list in element.find_all(['ul', 'ol'], recursive=True):
                nested_text = nested_list.get_text().strip()
                main_text = main_text.replace(nested_text, '').strip()

            result.append('-' * level + ' ' + main_text)
            for nested_list in element.find_all(['ul', 'ol'], recursive=False):
                result.extend(process_article(nested_list, level + 1))

        elif element.name == 'table':
            for row in element.find_all('tr'):
                cells = row.find_all(['th', 'td'])
                row_text = " | ".join([cell.get_text().strip() for cell in cells])
                result.append("| " + row_text + " |")

        elif hasattr(element, 'children'):
            for child in element.children:
                result.extend(process_article(child, level))

    return result
