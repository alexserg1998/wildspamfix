
## Quick setup

1. Клонируем репозиторий:
    ```bash
    git clone https://github.com/alexserg1998/wildspam
    ```
2. Соберем образ:
    ```bash
    docker compose build
   ```
3. Для того, чтобы запустить образ можем воспользоваться:
   ```bash
    docker compose up
   ```
5.  Далее можно перейти по адресу и попробовать 
   ```bash
    http://localhost:8000/docs
   ```
6. Так же можно из командной строки:
   ```bash
    python run.py -i data\\image.jpg
   ```
   или
   ```bash
    python run.py -i data - b 16
   ```
   * `--i` или `--image` это путь к изображению или папке с изображениями
   * `--b` или `--batch` это размер батча
   
Ответ мы получим в виде словаря:
   ``[
  {
    "data": "110948846.jpg",
    "predict": 0.07798244059085846
  },
  {
    "data": "123236586.jpg",
    "predict": 0.9603464603424072
  }
]``

6. Веса модели хранятся по ссылке:
   ```bash
    https://disk.yandex.ru/d/WuPwz_s86izWqg
   ```


