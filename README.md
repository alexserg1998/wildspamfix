## Быстрый старт

1. **Клонирование репозитория:**

    ```bash
    git clone https://github.com/alexserg1998/wildspam
    ```

2. **Сборка образа:**

    ```bash
    docker compose build
   ```

3. **Запуск образа:**

   ```bash
    docker compose up
   ```

4. **Переход по адресу:**

   ```bash
    http://localhost:8000/docs
   ```

5. **Пример запроса для `swin_predict_image`:**

    **Запрос:**
    ```json
    {
         "folder_path": "/work/data",
         "batch_size": 8,
         "threshold": 0.5
    }
    ```
    
    **Через curl:**
    ```bash
    curl -X 'POST' 'http://localhost:8000/swin_predict_image' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
       "folder_path": "/work/data",
       "batch_size": 8,
       "threshold": 0.5
    }'
    ```

    **Пример ответа:**
    ```json
    [
      {
        "data": "110948846.jpg",
        "predict": 0,
        "predict proba": 0.07798244059085846
      },
      {
        "data": "123236586.jpg",
        "predict": 1,
        "predict proba": 0.9603464603424072
      }
    ]
    ```

6. **Пример запроса для `swin_predict_files`:**

    **Через curl:**
    ```bash
      curl -X POST http://localhost:8000/swin_predict_files -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'files=@/work/data/110948846.jpg;type=image/jpeg' -F 'threshold=0.5'
    ```

    **Пример ответа:**
    ```json
    [
      {
        "data": "125835749.jpg",
        "predict": 1,
        "predict proba": 0.8902342915534973
      }
    ]
    ```

7. **Аргументы для `run.py`:**
   - `--i` или `--image`: путь к изображению или папке с изображениями
   - `--b` или `--batch`: размер батча
   - `--t` или `--threshold`: порог вероятности спама

8. **Ссылка на веса модели:**

    [Ссылка на веса модели](https://disk.yandex.ru/d/WuPwz_s86izWqg)

