import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import get_config

kb_cache = get_config()['KnowledgeBase']['cache']
kb_platrum_url = get_config()['KnowledgeBase']['knowledge_base_platrum_url']
kb_platrum_baseurl = get_config()['KnowledgeBase']['base_platrum_url']


def get_articles_url():
    html = get_html(kb_platrum_url)
    soup = BeautifulSoup(html, 'html.parser')

    data = {
        "total_articles": 0,
        "status_previous": 0,
        "status_new": 0,
        "status_deleted": 0,
        "status_updated": 0,
        "sections": {}
    }

    for section in soup.find_all('section', class_='cs-g article-list'):

        section_tag = section.find('div', class_='list-lead').find('a')
        section_name = section_tag['title']
        section_url = section_tag['href']

        article_tag = section_tag.find('span', class_='item-count')
        if article_tag:
            data['total_articles'] += int(article_tag.text)

        if section_url.startswith('/'):
            section_url = kb_platrum_baseurl + section_url

        parse_section(section_name, section_url, data)

        print(f"\rЗагрузка статей: сохранено {data['total_articles']}", end='')
        for i in range(5):
            print('.', end='', flush=True)
            time.sleep(0.1)

    new = save_to_cache(data)
    print('\r', end='')
    print(f"{data['total_articles']} статей успешно обработаны и сохранены в файл {kb_cache}")
    if new:
        print(f"Количество измений в статьях: {new}")


def parse_section(section_name, url, data):
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    section = soup.find('section', class_='article-list c-list')
    process_section(section_name, section, data)

    total_articles = 0
    for section_name, section in data['sections'].items():
        for _ in section['articles']:
            total_articles += 1

    pagination = soup.find('div', class_='pagination')
    if pagination:
        next_disabled = pagination.find('li', class_='next disabled')
        if not next_disabled:
            process_pagination(section_name, data, pagination, data)


def process_section(section_name, section, data):
    for article in section.find_all('div', class_='c-row c-article-row'):
        title_tag = article.find('div', class_='ellipsis article-title')
        url = title_tag.a['href']
        article_name = title_tag.a.text

        time_tag = article.find('div', class_='help-text')
        l_time = time_tag.text

        if url.startswith('/'):
            url = kb_platrum_baseurl + url

        article_id = re.search(r'/articles/(\d+)-', url)
        if article_id:
            article_id = article_id.group(1)
        else:
            article_id = None

        if section_name not in data['sections']:
            data["sections"][section_name] = {
                "articles": {}
            }

        data['sections'][section_name]['articles'][article_name] = {
            "id": article_id,
            "url": url,
            "updated_time": l_time,
            "status": 'new'
        }


def process_pagination(section_name, data, pagination, results):
    url = kb_platrum_baseurl + pagination.find('li', class_='next').find('a')['href']
    parse_section(section_name, url, data)


def get_html(url):
    try:
        response = requests.get(url, verify=False)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f'Ошибка запроса: {e}', file=sys.stderr)
        return ''


def save_to_cache(data):
    if data is None:
        print('Ошибка, нет данных для сохранения в файл')
        return

    try:
        with open(kb_cache, 'r', encoding='utf-8') as l_data_file:
            l_data = json.load(l_data_file)
    except FileNotFoundError:
        l_data = None

    count_mismatched = 0
    result_data = {
        "total_articles": 0,
        "status_previous": 0,
        "status_new": 0,
        "status_deleted": 0,
        "status_updated": 0,
        "sections": {}
    }

    if l_data:
        for section, section_data in l_data['sections'].items():

            if section not in result_data['sections']:
                result_data['sections'][section] = {"articles": {}}

            for article, article_data in section_data['articles'].items():

                if article in data['sections'].get(section, {}).get('articles', {}):
                    result_data['sections'][section]['articles'][article] = data['sections'][section]['articles'][
                        article]
                    if article_data['updated_time'] == data['sections'][section]['articles'][article]['updated_time']:
                        result_data['sections'][section]['articles'][article]['status'] = 'previous'
                        result_data['total_articles'] += 1
                        result_data['status_previous'] += 1
                    else:
                        result_data['sections'][section]['articles'][article]['status'] = 'updated'
                        result_data['total_articles'] += 1
                        result_data['status_updated'] += 1
                        count_mismatched += 1
                else:
                    result_data['sections'][section]['articles'][article] = article_data
                    result_data['sections'][section]['articles'][article]['status'] = 'deleted'
                    result_data['total_articles'] += 1
                    result_data['status_deleted'] += 1
                    count_mismatched += 1

        for section, section_data in data['sections'].items():
            if section not in result_data['sections']:
                result_data['sections'][section] = {"articles": {}}

            for article, article_data in section_data['articles'].items():
                if article not in result_data['sections'].get(section, {}).get('articles', {}):
                    result_data['sections'][section]['articles'][article] = article_data
                    result_data['sections'][section]['articles'][article]['status'] = 'new'
                    result_data['total_articles'] += 1
                    result_data['status_new'] += 1
                    count_mismatched += 1
    else:
        result_data = data
        result_data['status_new'] = data['total_articles']

    with open(kb_cache, 'w', encoding='utf-8') as data_file:
        json.dump(result_data, data_file, ensure_ascii=False, indent=4)

    return count_mismatched
