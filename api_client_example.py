"""
Пример клиента для взаимодействия с API отслеживания взгляда
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_api():
    """Тестирование всех эндпоинтов API"""
    print("=== Тестирование Gaze Tracking API ===\n")
    
    # Проверка статуса сервиса
    print("1. Проверка статуса сервиса:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")
        return
    
    # Получение текущего статуса
    print("2. Получение текущего статуса:")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {json.dumps(response.json(), indent=4)}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")
    
    # Запуск отслеживания
    print("3. Запуск отслеживания взгляда:")
    try:
        response = requests.post(f"{BASE_URL}/start", json={"camera_id": 0})
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")
    
    # Ожидание для сбора данных
    print("4. Ожидание 5 секунд для сбора данных...")
    time.sleep(5)
    
    # Получение координат взгляда
    print("5. Получение координат взгляда:")
    try:
        response = requests.get(f"{BASE_URL}/gaze")
        print(f"   Статус: {response.status_code}")
        gaze_data = response.json()
        print(f"   X: {gaze_data['x']:.3f}")
        print(f"   Y: {gaze_data['y']:.3f}")
        print(f"   Уверенность: {gaze_data['confidence']:.3f}")
        print(f"   Время: {gaze_data['timestamp']}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")
    
    # Получение статуса во время работы
    print("6. Получение статуса во время работы:")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {json.dumps(response.json(), indent=4)}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")
    
    # Остановка отслеживания
    print("7. Остановка отслеживания взгляда:")
    try:
        response = requests.post(f"{BASE_URL}/stop")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}\n")
    except Exception as e:
        print(f"   Ошибка: {e}\n")

def continuous_monitoring():
    """Непрерывный мониторинг координат взгляда"""
    print("=== Непрерывный мониторинг координат взгляда ===")
    print("Для остановки нажмите Ctrl+C\n")
    
    try:
        # Запуск отслеживания
        response = requests.post(f"{BASE_URL}/start", json={"camera_id": 0})
        if response.status_code == 200:
            print("Отслеживание запущено\n")
        else:
            print(f"Ошибка запуска: {response.text}")
            return
        
        while True:
            try:
                response = requests.get(f"{BASE_URL}/gaze")
                if response.status_code == 200:
                    gaze_data = response.json()
                    print(f"X: {gaze_data['x']:.3f}, Y: {gaze_data['y']:.3f}, "
                          f"Confidence: {gaze_data['confidence']:.3f}")
                else:
                    print(f"Ошибка получения данных: {response.status_code}")
                
                time.sleep(0.1)  # Обновление каждые 100ms
                
            except KeyboardInterrupt:
                print("\nОстановка мониторинга...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                time.sleep(1)
        
        # Остановка отслеживания
        response = requests.post(f"{BASE_URL}/stop")
        if response.status_code == 200:
            print("Отслеживание остановлено")
        else:
            print(f"Ошибка остановки: {response.text}")
            
    except Exception as e:
        print(f"Ошибка в continuous_monitoring: {e}")

if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1. Тестирование API")
    print("2. Непрерывный мониторинг")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        test_api()
    elif choice == "2":
        continuous_monitoring()
    else:
        print("Неверный выбор")