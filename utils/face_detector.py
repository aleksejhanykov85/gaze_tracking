"""
Утилиты для детектирования лица и глаз в реальном времени
"""
import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class FaceEyeDetector:
    """Детектор лица и глаз для реального времени"""
    
    def __init__(self):
        # Инициализация детектора dlib
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Загрузка модели для предсказания landmarks
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Трансформации для изображения глаз
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_face_and_eyes(self, frame):
        """Обнаружение лица и глаз на кадре"""
        # Конвертация в grayscale для детектора dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детектирование лиц
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None, None, None
        
        # Берем первое обнаруженное лицо
        face = faces[0]
        
        # Получение landmarks
        landmarks = self.predictor(gray, face)
        
        # Извлечение регионов глаз
        left_eye = self._extract_eye_region(frame, landmarks, "left")
        right_eye = self._extract_eye_region(frame, landmarks, "right")
        
        # Преобразование в PIL Image
        left_eye_pil = Image.fromarray(left_eye)
        right_eye_pil = Image.fromarray(right_eye)
        
        # Применение трансформаций
        left_eye_tensor = self.transform(left_eye_pil)
        right_eye_tensor = self.transform(right_eye_pil)
        
        # Объединение в batch (2 глаза)
        eyes_tensor = torch.stack([left_eye_tensor, right_eye_tensor])
        
        return eyes_tensor, face, landmarks
    
    def _extract_eye_region(self, frame, landmarks, side="left"):
        """Извлечение региона глаза по landmarks"""
        if side == "left":
            points = list(range(36, 42))
        else:  # right
            points = list(range(42, 48))
        
        # Получение координат глаза
        eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in points]
        
        # Определение bounding box
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Расширение bounding box
        padding = 10
        x_min = max(0, x_min - padding)
        x_max = x_max + padding
        y_min = max(0, y_min - padding)
        y_max = y_max + padding
        
        # Извлечение региона глаза
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Конвертация в RGB
        if len(eye_region.shape) == 2:
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_GRAY2RGB)
        else:
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        
        return eye_region
    
    def draw_gaze_vector(self, frame, eye_center, gaze_vector, length=50, color=(0, 255, 0)):
        """Рисование вектора взгляда на кадре"""
        # Масштабирование вектора
        gaze_vector_scaled = gaze_vector * length
        
        # Конечная точка вектора
        end_point = (
            int(eye_center[0] + gaze_vector_scaled[0]),
            int(eye_center[1] + gaze_vector_scaled[1])
        )
        
        # Рисование вектора
        cv2.arrowedLine(
            frame,
            (int(eye_center[0]), int(eye_center[1])),
            end_point,
            color,
            2,
            tipLength=0.3
        )
        
        # Рисование круга в центре глаза
        cv2.circle(frame, (int(eye_center[0]), int(eye_center[1])), 3, (0, 0, 255), -1)
        
        return frame

class GazeFilter:
    """Фильтр Калмана для сглаживания предсказаний взгляда"""
    
    def __init__(self, state_dim=4, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Инициализация фильтра Калмана
        self.kf = cv2.KalmanFilter(state_dim, measurement_dim)
        
        # Матрица перехода (предполагаем постоянную скорость)
        self.kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        if state_dim == 4 and measurement_dim == 2:
            # Для 4-мерного состояния (x, y, vx, vy) и 2-мерных измерений (x, y)
            self.kf.transitionMatrix[0, 2] = 1  # x += vx
            self.kf.transitionMatrix[1, 3] = 1  # y += vy
        
        # Матрица измерения
        if measurement_dim <= state_dim:
            self.kf.measurementMatrix = np.zeros((measurement_dim, state_dim), dtype=np.float32)
            for i in range(measurement_dim):
                self.kf.measurementMatrix[i, i] = 1
        
        # Ковариационные матрицы
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * 5e-2  # Увеличиваем шум процесса для большей чувствительности
        self.kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * 0.02  # Уменьшаем шум измерений
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32)
        
    def update(self, measurement):
        """Обновление фильтра с новым измерением"""
        # Предсказание
        self.kf.predict()
        
        # Коррекция
        if measurement is not None:
            measurement = np.array(measurement, dtype=np.float32).reshape(-1, 1)
            self.kf.correct(measurement)
        
        # Возвращаем текущее состояние (после коррекции)
        # Увеличиваем чувствительность к изменениям
        state = self.kf.statePost[:2].flatten()
        # Центрируем значения относительно 0.5 и увеличиваем чувствительность
        centered_state = (state - 0.5) * 1.1 + 0.5  # Увеличиваем чувствительность на 10%
        # Ограничиваем значения в пределах [0, 1]
        centered_state = np.clip(centered_state, 0, 1)
        return centered_state