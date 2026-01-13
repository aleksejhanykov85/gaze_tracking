"""
Скрипт для загрузки shape_predictor_68_face_landmarks.dat
"""
import os
import requests
import bz2

def download_shape_predictor():
    """Загрузка shape_predictor_68_face_landmarks.dat"""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_filename = "shape_predictor_68_face_landmarks.dat"
    
    print("Загрузка shape_predictor_68_face_landmarks.dat...")
    
    try:
        # Загрузка файла с помощью requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Файл {filename} успешно загружен")
        
        # Распаковка
        print("Распаковка файла...")
        with bz2.BZ2File(filename) as f_in:
            data = f_in.read()
            with open(extracted_filename, 'wb') as f_out:
                f_out.write(data)
        
        print(f"Файл {extracted_filename} успешно создан")
        
        # Удаление архива
        os.remove(filename)
        print(f"Временный файл {filename} удален")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при загрузке или распаковке: {e}")
        return False

if __name__ == "__main__":
    if download_shape_predictor():
        print("\nЗагрузка завершена! Теперь можно запустить realtime_gaze.py")
    else:
        print("\nОшибка при загрузке. Попробуйте вручную загрузить файл с http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")