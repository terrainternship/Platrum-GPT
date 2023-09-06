import re

import requests


def download_document_text(url: str) -> str:
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    return response.text


'''
FILE_ID = '1xUCQxlzLYalb1FTt1r24P0cLJVk9AYonV_zPv4z5QrQ'

file_path = 'result.docx'
share_link = 'https://docs.google.com/document/d/1xUCQxlzLYalb1FTt1r24P0cLJVk9AYonV_zPv4z5QrQ/edit?usp=sharing'

with open('result.docx', 'rb') as f:
  files = {'data': f}
  response = requests.post(share_link, files=files)
'''
