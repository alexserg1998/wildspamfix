import argparse
import os
import requests


def predict_image(image_path):
    files = {'files': open(image_path, 'rb')}
    response = requests.post("http://0.0.0.0:8000/swin_predict_files", files=files)
    print(response.json())


def predict_folder(folder_path, batch_size):
    data = {"folder_path": folder_path, "batch_size": batch_size}
    response = requests.post("http://0.0.0.0:8000/swin_predict_image", json=data)
    print(response.json())


def main():
    parser = argparse.ArgumentParser(description='Предсказание спама на изображениях с помощью Swin Transformer')
    parser.add_argument('-i', '--input', type=str, help='Путь к изображению или папке с изображениями')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Размер пакета (для папки)')
    args = parser.parse_args()

    if args.input:
        if os.path.isfile(args.input):
            predict_image(args.input)
        elif os.path.isdir(args.input):
            predict_folder(args.input, args.batch_size)
        else:
            print("Указанный путь не является файлом или папкой.")


if __name__ == "__main__":
    main()
