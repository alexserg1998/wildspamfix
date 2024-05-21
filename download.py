import requests
from tqdm import tqdm
from urllib.parse import urlencode
import zipfile
import os

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/WuPwz_s86izWqg'
name_file = 'weight.zip'

final_url = base_url + urlencode({'public_key': public_key})
response = requests.get(final_url)
download_url = response.json()['href']

if response.status_code == 200:
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # Размер блока для отслеживания прогресса
        progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)

        with open(name_file, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        print("File successfully downloaded.")
        # Разархивирование с помощью модуля zipfile
        with zipfile.ZipFile(name_file, 'r') as zip_ref:
            zip_ref.extractall()

        # Удаляем скачанный zip-файл после разархивирования
        os.remove(name_file)
        print("File successfully zipfile.")
    else:
        print("Failed to download the file. Please check the URL link and file access.")
else:
    print("Failed to obtain download link.")
