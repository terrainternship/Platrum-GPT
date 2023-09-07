import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import html2text
import re

images_set = set()
videos_set = set()
tables_set = set()

# токен для коротких адресов https://goo.su
api_token = 'CK9VGp4hLMNRqpH46Oa2chAeT0CQiSrInEFvohbTahtt0yXlyoyKZ8fwue8U'

# Функция для обработки страницы с текстом
def process_text_page(url, page_number, output_file):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверяем, был ли успешный ответ сервера
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        # Извлекаем title страницы
        title = soup.find('title').text.strip()

        # Создаем множество для отслеживания записанных заголовков
        written_headings = set()

        # Открываем файл для записи
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\t#{title}\n")

            # Обрабатываем элемент article (всё, что нас интересует на этом конкретном сайте, расположено в этом элементе)
            article = soup.find('article', class_='article-body')
            if article:
                # Выбираем элементы, которые нас интересуют. в iframe хранятся ссылки на YouTube
                elements = article.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'img', 'iframe', 'a', 'table'])
                for element in elements:
                    if element.name.startswith('h'):
                        level = element.name[1:]
                        level = int(level) +1
                        text = element.get_text(strip=True)
                        if text and text not in written_headings:
                            # задаем хеши в зависимости от уровня заголовка 
                            f.write(f"\n{'#' * int(level)}{text.strip()}\n\n")
                            written_headings.add(text)
                    elif element.name == 'p':
                        text = element.get_text(strip=True)
                        # игнорируем абзац "СОДЕРЖАНИЕ"
                        if text and text != "СОДЕРЖАНИЕ":
                            # Исправляем склеивание после точки и перед ссылкой
                            text = re.sub(r'(?<=[а-яА-Яa-zA-Z])\.(?=[а-яА-Яa-zA-Z])', '. ', text)
                            text = re.sub(r'(?<=[а-яА-Яa-zA-Z])https?://', ' https://', text)
                            f.write(f"\n{text}\n")
                    elif element.name == 'img':
                        img_src = element.get('src')
                        if img_src not in images_set:
                            images_set.add(img_src)
                        if img_src:
                            link = img_src
                            # Здесь выполняем запрос к сервису сокращения ссылок
                            response = requests.post('https://goo.su/api/links/create', 
                                         headers={'x-goo-api-token': api_token},
                                         data={'url': link})
                            if response.status_code == 200:
                                short_link = response.json().get('short_url')
                                # записываем в файл в формате См. Рисунок №1: https://goo.su/XXXXX
                                f.write(f"См. Рисунок {len(images_set)}: {short_link}\n")
                            else:
                                # если сервис https://goo.su недоступен, пишем оригинальную ссылку в том же формате 
                                print("response.status_code = ", response.status_code)
                                f.write(f"См. Рисунок {len(images_set)}: {img_src}\n")
                    elif element.name == 'iframe':
                        # так как на сайте видео оформлено через iframe, то извлекаем ссылки из него
                        iframe_src = element.get('src')
                        if iframe_src and "youtube.com/embed/" in iframe_src:
                            video_id = urlparse(iframe_src).path.split('/')[-1]
                            video_link = f"https://www.youtube.com/watch?v={video_id}"
                            if video_link not in videos_set:
                                videos_set.add(video_link)
                                link = video_link
                                # Здесь выполняем запрос к сервису сокращения ссылок
                                response = requests.post('https://goo.su/api/links/create', 
                                         headers={'x-goo-api-token': api_token},
                                         data={'url': link})
                                if response.status_code == 200:
                                    short_link = response.json().get('short_url')
                                    print("Видео = ", short_link)
                                    # записываем в файл в формате См. Видео №1: https://goo.su/XXXXX
                                    f.write(f"См. Видео {len(videos_set)}: {short_link}\n")
                                else:
                                    print("response.status_code = ", response.status_code)
                                    # если сервис https://goo.su недоступен, пишем оригинальную ссылку в том же формате
                                    f.write(f"См. Видео {len(videos_set)}: {video_link}\n")
                    elif element.name == 'a' and not element['href'].startswith('#')  and not element['href'] == 'https://platrum.ru/': # здесь оформляем ссылки на статьи
                        text = element.get_text(strip=True)
                        link = element['href']
                        # Здесь выполняем запрос к сервису сокращения ссылок
                        response = requests.post('https://goo.su/api/links/create', 
                                         headers={'x-goo-api-token': api_token},
                                         data={'url': link})
                        if response.status_code == 200:
                            short_link = response.json().get('short_url')
                            print("см. статью ", short_link)
                            f.write(f"{text} - см. статью: {short_link}\n")
                        else:
                            f.write(f"{text} - см. статью: {link}\n")
                    elif element.name == 'table': # обработка таблиц
                        # Находим все строки (теги <tr>) в таблице
                        rows = element.find_all('tr')
                        if rows:
                            table_number = len(tables_set) + 1
                            tables_set.add(table_number)
                            # Выводим заголовок таблицы
                            f.write(f"\nТаблица {table_number}\n")
                            # Обрабатываем каждую строку
                            for row in rows:
                                cells = row.find_all(['th', 'td'])  # Ищем ячейки в строке
                                if cells:
                                    row_data = "|".join(cell.get_text(strip=True) for cell in cells)
                                    f.write(f"-| {row_data} |\n")
    # обработка ошибок
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while processing {url}: {e}")

# Стартовая страница для парсинга
start_url = 'https://help.platrum.ru/support/home'

# Путь к файлу для записи текста
output_file = 'parsed.txt'

# Парсинг начинаем с каталогов (все статьи на этом сайте лежат в каталоге folders и далее по каталогам)
response = requests.get(start_url)
soup = BeautifulSoup(response.text, 'html.parser')
folder_links = [urljoin(start_url, link['href']) for link in soup.find_all('a', href=True) if link['href'].startswith('/support/solutions/folders/')]

# Создаем множество для отслеживания посещенных URL (чтобы не ходить по перекрестным ссылкам)
visited_pages = set()
page_number = 1

# Создаем множество для отслеживания посещенных каталогов (чтобы не ходить два раза в один каталог)
visited_folders = set()

# Парсим каждую страницу каталога
for folder_link in folder_links:
    if folder_link in visited_folders:
        continue
    visited_folders.add(folder_link)
    
    print("Каталог: ", folder_link) # выводим информацию о текущей папке
    folder_response = requests.get(folder_link)
    folder_soup = BeautifulSoup(folder_response.text, 'html.parser')
    article_links = [urljoin(folder_link, link['href']) for link in folder_soup.find_all('a', href=True) if link['href'].startswith('/support/solutions/articles/')]
    
    for article_link in article_links:
        print("Страница: ", page_number)  # выводим информацию о текущей странице
        process_text_page(article_link, page_number, output_file)
        visited_pages.add(article_link)
        page_number += 1

# Выводим информацию о количестве обработанных страниц
print(f"Парсинг завершен. Всего обработано страниц: {len(visited_pages)}")
print(f"Всего рисунков в файле: <{len(images_set)}>\n")
print(f"Всего видео в файле: <{len(videos_set)}>\n")
print(f"Всего таблиц в файле: <{len(tables_set)}>\n")
