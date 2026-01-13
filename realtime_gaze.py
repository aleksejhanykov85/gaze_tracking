"""
Скрипт для отслеживания взгляда в реальном времени с веб-камеры
"""
import cv2
import torch
import numpy as np
import time
from config import Config
from models.gaze_model import GazeTrackingModel
from utils.face_detector import FaceEyeDetector, GazeFilter

class RealTimeGazeTracker:
    """Класс для отслеживания взгляда в реальном времени"""
    
    def __init__(self, model_path=None):
        # Конфигурация
        self.config = Config()
        self.device = self.config.get_device()
        
        # Загрузка модели
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Инициализация детектора
        self.detector = FaceEyeDetector()
        
        # Инициализация фильтра Калмана
        self.gaze_filter = GazeFilter()
        
        # Статистика
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # История взглядов
        self.gaze_history = []
        self.max_history = 30
        
        # Цвета для визуализации
        self.colors = {
            'gaze': (0, 255, 0),      # Зеленый
            'face': (255, 0, 0),      # Синий
            'eyes': (0, 0, 255),      # Красный
            'text': (255, 255, 255)   # Белый
        }
    
    def _load_model(self, model_path):
        """Загрузка обученной модели"""
        if model_path is None:
            model_path = self.config.model_save_path
        
        print(f"Загрузка модели из: {model_path}")
        
        # Создание модели
        model = GazeTrackingModel(
            backbone=self.config.backbone,
            num_gaze_points=self.config.num_gaze_points,
            dropout_rate=self.config.dropout_rate,
            pretrained=False
        ).to(self.device)
        
        # Загрузка весов
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Модель успешно загружена")
        return model
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        # Детектирование лица и глаз
        eyes_tensor, face, landmarks = self.detector.detect_face_and_eyes(frame)
        
        if eyes_tensor is None:
            # Если глаза не обнаружены, обновляем только FPS и возвращаем кадр
            self._update_fps()
            # Визуализируем без предсказания взгляда
            frame = self._visualize_results(frame, face, landmarks, None, raw_prediction=None)
            return frame
        
        # Предсказание взгляда
        with torch.no_grad():
            eyes_tensor = eyes_tensor.to(self.device)
            # Предсказание для каждого глаза
            predictions = self.model(eyes_tensor)
            
            # Обработка предсказаний для левого и правого глаза
            left_eye_pred = predictions[0].cpu().numpy()
            right_eye_pred = predictions[1].cpu().numpy()
            
            # Усреднение предсказаний для двух глаз с учетом уверенности
            avg_prediction = (left_eye_pred + right_eye_pred) / 2.0
            
        
        # Используем фильтр Калмана для сглаживания предсказаний
        filtered_gaze = self.gaze_filter.update(avg_prediction)
        
        # Увеличиваем чувствительность к изменениям взгляда
        # Преобразуем координаты так, чтобы они были более чувствительными к изменениям
        # Центрируем значения относительно 0.5 (середины диапазона [0, 1])
        centered_gaze = (filtered_gaze - 0.5) * 1.2 + 0.5  # Увеличиваем чувствительность на 20%
        # Ограничиваем значения в пределах [0, 1]
        centered_gaze = np.clip(centered_gaze, 0, 1)
        filtered_gaze = centered_gaze
        
        
        
        # Сохранение в историю
        self.gaze_history.append(filtered_gaze.copy())
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        # Визуализация результатов
        frame = self._visualize_results(frame, face, landmarks, filtered_gaze, raw_prediction=avg_prediction)
        
        # Обновление статистики FPS
        self._update_fps()
        
        return frame
    
    def _visualize_results(self, frame, face, landmarks, gaze_vector, raw_prediction=None):
        """Визуализация результатов на кадре"""
        # Рисование bounding box лица
        if face is not None:
            cv2.rectangle(frame,
                         (face.left(), face.top()),
                         (face.right(), face.bottom()),
                         self.colors['face'], 2)
        
        # Рисование landmarks глаз
        if landmarks is not None:
            for i in range(36, 48):  # Только точки глаз
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, self.colors['eyes'], -1)
        
        # Рисование точки, куда смотрит пользователь (предсказание взгляда на экране)
        # Преобразуем нормализованные координаты взгляда в координаты экрана
        # gaze_vector содержит значения от 0 до 1 из-за Sigmoid функции активации, преобразуем их в координаты экрана
        # Преобразуем из [0, 1] в [0, frame_width] и [0, frame_height]
        if gaze_vector is not None:
            screen_x = int(gaze_vector[0] * frame.shape[1])  # Преобразуем из [0, 1] в [0, frame_width]
            screen_y = int(gaze_vector[1] * frame.shape[0])  # Преобразуем из [0, 1] в [0, frame_height]
            
            # Ограничение координат в пределах кадра
            screen_x = max(0, min(frame.shape[1], screen_x))
            screen_y = max(0, min(frame.shape[0], screen_y))
            
            # Рисуем точку взгляда
            cv2.circle(frame, (screen_x, screen_y), 10, (0, 255, 255), -1)  # Желтый круг
            cv2.circle(frame, (screen_x, screen_y), 12, (0, 0, 255), 2)      # Красная окантовка
            
            # Добавим текст с координатами точки взгляда
            gaze_coords_text = f"Gaze: ({screen_x}, {screen_y})"
            cv2.putText(frame, gaze_coords_text, (screen_x + 15, screen_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                 
        # Добавление текстовой информации
        # Передаем raw_prediction, которое может быть передано извне
        self._add_text_overlay(frame, gaze_vector, raw_prediction=raw_prediction)
        
        # Добавление тепловой карты истории взглядов
        if len(self.gaze_history) > 5:
            frame = self._add_gaze_heatmap(frame)
        
        return frame
    
    def _add_text_overlay(self, frame, gaze_vector, raw_prediction=None):
        """Добавление текстовой информации на кадр"""
        # Отображение координат взгляда, если они доступны
        if gaze_vector is not None:
            # Координаты взгляда (значения от 0 до 1 из-за Sigmoid функции активации)
            gaze_text = f"Gaze: ({gaze_vector[0]:.2f}, {gaze_vector[1]:.2f}) [0-1 range]"
            cv2.putText(frame, gaze_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
                       
            # Отображение необработанных предсказаний для диагностики
            if raw_prediction is not None:
                raw_text = f"Raw: ({raw_prediction[0]:.2f}, {raw_prediction[1]:.2f})"
                cv2.putText(frame, raw_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        else:
            # Если взгляда нет, отображаем сообщение об этом
            no_gaze_text = "Gaze: Not detected"
            cv2.putText(frame, no_gaze_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Подсказки
        info_text = "Press 'q' to quit, 's' to save frame"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def _add_gaze_heatmap(self, frame):
        """Добавление тепловой карты скоплений взглядов"""
        # Создание тепловой карты
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for gaze in self.gaze_history[-10:]:  # Последние 10 точек для улучшения производительности
            # gaze содержит значения в диапазоне [0, 1] из-за Sigmoid функции активации, преобразуем их в координаты экрана
            x = int(gaze[0] * frame.shape[1])  # Преобразуем из [0, 1] в [0, frame_width]
            y = int(gaze[1] * frame.shape[0])  # Преобразуем из [0, 1] в [0, frame_height]
            
            # Проверяем, что координаты находятся в пределах кадра
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Уменьшаем радиус для улучшения производительности
                x_min, x_max = max(0, x-10), min(frame.shape[1], x+10)
                y_min, y_max = max(0, y-10), min(frame.shape[0], y+10)
                
                # Проверяем, что диапазоны не пустые
                if x_max > x_min and y_max > y_min:
                    # Создаем сетку координат для текущего региона
                    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
                    # Вычисляем расстояния
                    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
                    # Применяем Гауссово размытие только к региону
                    mask = distances < 10
                    heatmap[y_min:y_max, x_min:x_max] += mask * np.exp(-distances**2 / (2 * 5**2))  # Уменьшаем sigma для улучшения производительности
        
        # Нормализация
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = (heatmap / max_val) * 255
        heatmap = heatmap.astype(np.uint8)
        
        # Применение цветовой карты
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Наложение на оригинальный кадр с прозрачностью
        alpha = 0.3
        frame = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return frame
    
    def _update_fps(self):
        """Обновление статистики FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:  # Каждую секунду
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Запуск основного цикла обработки видео"""
        # Инициализация видеозахвата
        cap = cv2.VideoCapture(self.config.camera_id)
        
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру {self.config.camera_id}")
            return
        
        print("\nЗапуск отслеживания взгляда в реальном времени...")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' для сохранения кадра")
        
        frame_count = 0
        saved_frames = 0
        
        while True:
            # Захват кадра
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: не удалось захватить кадр")
                break
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Отображение результата
            cv2.imshow('Gaze Tracking', processed_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Завершение работы...")
                break
            elif key == ord('s'):
                # Сохранение кадра
                filename = f"gaze_frame_{saved_frames:04d}.png"
                cv2.imwrite(filename, processed_frame)
                print(f"Кадр сохранен как {filename}")
                saved_frames += 1
            elif key == ord('h'):
                # Переключение тепловой карты
                self.max_history = 0 if self.max_history > 0 else 30
                print(f"Тепловая карта: {'выключена' if self.max_history == 0 else 'включена'}")
            
            frame_count += 1
        
        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработано кадров: {frame_count}")
        print(f"Средний FPS: {self.fps:.1f}")

def main():
    """Основная функция для запуска в реальном времени"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Отслеживание взгляда в реальном времени')
    parser.add_argument('--model', type=str, default=None,
                       help='Путь к обученной модели')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID камеры (по умолчанию 0)')
    
    args = parser.parse_args()
    
    # Запуск трекера
    tracker = RealTimeGazeTracker(model_path=args.model)
    
    if args.camera != 0:
        tracker.config.camera_id = args.camera
    
    tracker.run()

if __name__ == "__main__":
    main()