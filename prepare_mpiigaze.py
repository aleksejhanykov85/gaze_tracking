
import scipy.io
import os
import pandas as pd
from shutil import copyfile
import numpy as np

def convert_mpiigaze_to_csv(dataset_root='./data/MPIIGaze', output_csv='./data/MPIIGaze/annotations.csv'):
    """
    Конвертирует исходные данные MPIIGaze в CSV-файл для вашего проекта.
    """
    base_original_path = os.path.join(dataset_root, 'Data', 'Original')
    images_output_dir = os.path.join(dataset_root, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Создаем папку для CSV файла, если она не существует
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    annotations_data = []

    # Проходим по всем папкам участников (p00, p01, ..., p14)
    for participant_dir in sorted(os.listdir(base_original_path)):
        participant_path = os.path.join(base_original_path, participant_dir)
        if not os.path.isdir(participant_path):
            continue

        print(f"Обрабатываю участника: {participant_dir}")

        # Проходим по всем папкам дней внутри папки участника
        for day_dir in sorted(os.listdir(participant_path)):
            day_path = os.path.join(participant_path, day_dir)
            if not os.path.isdir(day_path):
                continue

            annotation_file = os.path.join(day_path, 'annotation.txt')
            if not os.path.exists(annotation_file):
                print(f"  Предупреждение: файл аннотаций не найден {annotation_file}")
                continue

            # Читаем файл аннотаций
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            print(f"  День {day_dir}: обрабатываю {len(lines)} записей")

            # Получаем список файлов в папке дня
            available_files = os.listdir(day_path)
            
            # Каждая строка соответствует одному изображению
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 41:  # Проверяем, что в строке достаточно данных
                    continue

                # Имя изображения - это числовой индекс (например, '570')
                # В датасете изображения имеют формат: 0001.jpg, 0002.jpg и т.д.
                image_number = parts[0]
                
                # Преобразуем номер в формат с 4 цифрами с ведущими нулями
                try:
                    image_num_int = int(image_number)
                    image_name = f"{image_num_int:04d}.jpg"
                except ValueError:
                    # Если не число, пробуем как есть
                    image_name = image_number
                    if not image_name.endswith('.jpg'):
                        image_name += '.jpg'

                # Полный путь к исходному изображению в датасете
                source_image_path = os.path.join(day_path, image_name)

                # Новый путь для копии в папке images/
                # Создаём уникальное имя, чтобы избежать конфликтов
                unique_image_name = f"{participant_dir}_{day_dir}_{image_name}"
                dest_image_path = os.path.join(images_output_dir, unique_image_name)

                # Проверяем, существует ли файл изображения
                if not os.path.exists(source_image_path):
                    # Пробуем найти файл с другим расширением или форматом
                    found = False
                    for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                        alt_name = f"{image_num_int:04d}{ext}"
                        alt_path = os.path.join(day_path, alt_name)
                        if os.path.exists(alt_path):
                            source_image_path = alt_path
                            found = True
                            break
                    
                    if not found:
                        # Пробуем найти файл по исходному имени из списка файлов
                        for file in available_files:
                            if file.startswith(image_number) and file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                                source_image_path = os.path.join(day_path, file)
                                found = True
                                break
                    
                    if not found:
                        print(f"  Предупреждение: файл не найден {image_name} в {day_path}")
                        continue

                # Копируем изображение в новую структуру
                if os.path.exists(source_image_path):
                    copyfile(source_image_path, dest_image_path)
                else:
                    print(f"  Предупреждение: файл не найден {source_image_path}")
                    continue

                # Извлекаем координаты цели взгляда на экране
                try:
                    screen_gaze_x = float(parts[24])  # Индекс 24 соответствует Dimension 25
                    screen_gaze_y = float(parts[25])  # Индекс 25 соответствует Dimension 26
                except (ValueError, IndexError):
                    print(f"  Ошибка: неверный формат координат в строке {line_idx}")
                    continue

                # Нормализуем координаты экрана к диапазону [0, 1]
                # В MPIIGaze разрешение экрана 1920x1080 пикселей
                screen_width_px = 1920.0
                screen_height_px = 1080.0

                normalized_gaze_x = screen_gaze_x / screen_width_px
                normalized_gaze_y = screen_gaze_y / screen_height_px

                # Добавляем запись в список аннотаций
                annotations_data.append({
                    'image_path': os.path.join('images', unique_image_name).replace('\\', '/'),
                    'original_path': os.path.join(participant_dir, day_dir, image_name).replace('\\', '/'),
                    'participant': participant_dir,
                    'day': day_dir,
                    'image_number': image_number,
                    'gaze_x': normalized_gaze_x,
                    'gaze_y': normalized_gaze_y,
                    'screen_gaze_x': screen_gaze_x,
                    'screen_gaze_y': screen_gaze_y
                })

    # Создаём DataFrame и сохраняем в CSV
    if annotations_data:
        df = pd.DataFrame(annotations_data)
        df.to_csv(output_csv, index=False)
        print(f"Аннотации успешно сохранены в {output_csv}")
        print(f"Всего обработано записей: {len(df)}")
        print(f"Размер DataFrame: {df.shape}")
        
        # Выводим статистику
        print(f"\nСтатистика по участникам:")
        print(df['participant'].value_counts().sort_index())
    else:
        print("Ошибка: не было обработано ни одной записи!")

if __name__ == "__main__":
    # Предполагаем, что датасет распакован в папку 'data/mpiigaze' внутри проекта
    dataset_path = r'C:\Users\user\Desktop\profv2\data\MPIIGaze'
    
    # Проверяем существование пути
    if not os.path.exists(dataset_path):
        print(f"Ошибка: путь {dataset_path} не существует!")
        print("Проверьте, правильно ли указан путь к датасету.")
        exit(1)
    
    # Выводим структуру датасета для отладки
    print("Структура датасета:")
    data_original_path = os.path.join(dataset_path, 'Data', 'Original')
    if os.path.exists(data_original_path):
        participants = os.listdir(data_original_path)
        print(f"Найдено участников: {len(participants)}")
        for p in sorted(participants)[:5]:  # Показываем первые 5 для примера
            p_path = os.path.join(data_original_path, p)
            if os.path.isdir(p_path):
                days = os.listdir(p_path)
                print(f"  {p}: дней - {len(days)}")
    else:
        print(f"Ошибка: не найден путь {data_original_path}")
        print("Возможно, датасет имеет другую структуру.")
    
    convert_mpiigaze_to_csv(dataset_root=dataset_path)
