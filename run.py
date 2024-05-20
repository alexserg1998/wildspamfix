import argparse
import os
import requests
from typing import Any, Dict


def predict_image(image_path: str, threshold: float) -> Dict[str, Any]:
    with open(image_path, 'rb') as image_file:
        files = {'files': image_file}
        data = {'threshold': threshold}
        response = requests.post("http://127.0.0.1:8000/swin_predict_files", files=files, data=data)
    return response.json()


def predict_folder(folder_path: str, batch_size: int, threshold: float) -> Dict[str, Any]:
    data = {"folder_path": folder_path, "batch_size": batch_size, "threshold": threshold}
    response = requests.post("http://127.0.0.1:8000/swin_predict_image", json=data)
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description='Предсказание спама на изображениях с помощью Swin Transformer')
    parser.add_argument('-i', '--input', type=str, help='Путь к изображению или папке с изображениями')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Размер пакета (для папки)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Порог вероятности спама')
    args = parser.parse_args()

    if args.input:
        if os.path.isfile(args.input):
            result = predict_image(args.input, args.threshold)
            print(result)
        elif os.path.isdir(args.input):
            result = predict_folder(args.input, args.batch_size, args.threshold)
            print(result)
        else:
            print("Указанный путь не является файлом или папкой.")


if __name__ == "__main__":
    main()
