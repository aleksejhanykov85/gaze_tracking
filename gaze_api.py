"""
FastAPI сервер для системы отслеживания взгляда
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import threading
import time
import cv2
import torch
import numpy as np
from config import Config
from models.gaze_model import GazeTrackingModel
from utils.face_detector import FaceEyeDetector, GazeFilter
import json
import os
import uvicorn

app = FastAPI(
    title="Gaze Tracking API",
    description="API для системы отслеживания взгляда в реальном времени",
    version="1.0.0"
)

class GazeTracker:
    """Класс для отслеживания взгляда с поддержкой многопоточности"""
    
    def __init__(self):
        self.config = Config()
        self.device = self.config.get_device()
        
        # Загрузка модели
        self.model = self._load_model()
        self.model.eval()
        
        # Инициализация детектора
        self.detector = FaceEyeDetector()
        
        # Инициализация фильтра Калмана
        self.gaze_filter = GazeFilter()
        
        # Состояние отслеживания
        self.is_running = False
        self.current_gaze = {"x": 0.5, "y": 0.5, "confidence": 0.0}
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Поток для обработки видео
        self.video_thread = None
        self.cap = None
        
    def _load_model(self, model_path=None):
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
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Модель успешно загружена")
        except Exception as e:
            print(f"Не удалось загрузить модель: {e}")
            print("Используется новая модель")
        
        return model
    
    def _process_frame(self, frame):
        """Обработка одного кадра"""
        # Детектирование лица и глаз
        eyes_tensor, face, landmarks = self.detector.detect_face_and_eyes(frame)
        
        if eyes_tensor is None:
            # Если глаза не обнаружены, обновляем только FPS
            self._update_fps()
            return None
        
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
            centered_gaze = (filtered_gaze - 0.5) * 1.2 + 0.5  # Увеличиваем чувствительность на 20%
            # Ограничиваем значения в пределах [0, 1]
            centered_gaze = np.clip(centered_gaze, 0, 1)
            filtered_gaze = centered_gaze
            
            # Обновляем текущие координаты взгляда
            self.current_gaze = {
                "x": float(filtered_gaze[0]),
                "y": float(filtered_gaze[1]),
                "confidence": float(np.mean(np.abs(avg_prediction - filtered_gaze)))
            }
        
        # Обновление статистики FPS
        self._update_fps()
        
        return self.current_gaze
    
    def _update_fps(self):
        """Обновление статистики FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:  # Каждую секунду
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def _run_video_processing(self):
        """Основной цикл обработки видео"""
        self.cap = cv2.VideoCapture(self.config.camera_id)
        
        if not self.cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру {self.config.camera_id}")
            self.is_running = False
            return
        
        print("\nЗапуск отслеживания взгляда в реальном времени...")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка: не удалось захватить кадр")
                break
            
            # Обработка кадра
            self._process_frame(frame)
        
        # Освобождение ресурсов
        if self.cap:
            self.cap.release()
        
        print("Отслеживание взгляда остановлено")

    def start_tracking(self):
        """Запуск отслеживания взгляда"""
        if self.is_running:
            return False
        
        self.is_running = True
        self.video_thread = threading.Thread(target=self._run_video_processing)
        self.video_thread.start()
        return True
    
    def stop_tracking(self):
        """Остановка отслеживания взгляда"""
        if not self.is_running:
            return False
        
        self.is_running = False
        if self.video_thread:
            self.video_thread.join()
        
        if self.cap:
            self.cap.release()
        
        return True
    
    def get_current_gaze(self):
        """Получение текущих координат взгляда"""
        return self.current_gaze.copy()
    
    def get_status(self):
        """Получение статуса отслеживания"""
        return {
            "is_running": self.is_running,
            "fps": self.fps,
            "current_gaze": self.current_gaze
        }

# Глобальный экземпляр трекера
gaze_tracker = GazeTracker()


class TrackingRequest(BaseModel):
    """Модель запроса для управления отслеживанием"""
    camera_id: Optional[int] = 0


class GazeResponse(BaseModel):
    """Модель ответа с координатами взгляда"""
    x: float
    y: float
    confidence: float
    timestamp: float


class StatusResponse(BaseModel):
    """Модель ответа со статусом системы"""
    is_running: bool
    fps: float
    current_gaze: Dict[str, float]
    timestamp: float


@app.get("/", response_class=HTMLResponse)
async def root():
    """Корневой эндпоинт с информационной страницей"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gaze Tracking API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .info-box {
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .endpoint {
                background-color: #f0f0f0;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #007acc;
            }
            a {
                color: #007acc;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👁️ Gaze Tracking API</h1>
            <p>Добро пожаловать в API для системы отслеживания взгляда в реальном времени!</p>
            
            <div class="info-box">
                <strong>Версия:</strong> 1.0.0<br>
                <strong>Статус:</strong> Работает<br>
                <strong>Автор:</strong> Gaze Tracking System
            </div>
            
            <h2>Доступные эндпоинты:</h2>
            <div class="endpoint">
                <a href="/docs"><strong>/docs</strong></a> - Интерактивная документация API (Swagger UI)
            </div>
            <div class="endpoint">
                <a href="/redoc"><strong>/redoc</strong></a> - Альтернативная документация (ReDoc)
            </div>
            <div class="endpoint">
                <strong>/health</strong> - Проверка состояния сервиса
            </div>
            <div class="endpoint">
                <strong>/status</strong> - Получить статус системы отслеживания
            </div>
            <div class="endpoint">
                <strong>/gaze</strong> - Получить текущие координаты взгляда
            </div>
            <div class="endpoint">
                <strong>/start</strong> - Запустить отслеживание взгляда (POST)
            </div>
            <div class="endpoint">
                <strong>/stop</strong> - Остановить отслеживание взгляда (POST)
            </div>
            
            <h2>Использование:</h2>
            <p>Для начала работы с API:</p>
            <ol>
                <li>Перейдите в <a href="/docs">документацию</a> для просмотра всех эндпоинтов</li>
                <li>Запустите отслеживание с помощью POST запроса к <strong>/start</strong></li>
                <li>Получайте координаты взгляда через GET запрос к <strong>/gaze</strong></li>
            </ol>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Обработка запроса favicon для избежания ошибок 404"""
    return HTMLResponse(content="", status_code=204)


@app.post("/start", summary="Запустить отслеживание взгляда")
async def start_tracking(request: Optional[TrackingRequest] = None):
    """Запуск отслеживания взгляда в реальном времени"""
    if request:
        gaze_tracker.config.camera_id = request.camera_id
    
    success = gaze_tracker.start_tracking()
    
    if success:
        return {
            "message": "Отслеживание взгляда запущено",
            "camera_id": gaze_tracker.config.camera_id,
            "timestamp": time.time()
        }
    else:
        raise HTTPException(status_code=400, detail="Отслеживание уже запущено")


@app.post("/stop", summary="Остановить отслеживание взгляда")
async def stop_tracking():
    """Остановка отслеживания взгляда"""
    success = gaze_tracker.stop_tracking()
    
    if success:
        return {
            "message": "Отслеживание взгляда остановлено",
            "timestamp": time.time()
        }
    else:
        raise HTTPException(status_code=400, detail="Отслеживание не запущено")


@app.get("/gaze", response_model=GazeResponse, summary="Получить текущие координаты взгляда")
async def get_gaze():
    """Получение текущих координат взгляда"""
    gaze_data = gaze_tracker.get_current_gaze()
    
    return GazeResponse(
        x=gaze_data["x"],
        y=gaze_data["y"],
        confidence=gaze_data["confidence"],
        timestamp=time.time()
    )


@app.get("/status", response_model=StatusResponse, summary="Получить статус системы")
async def get_status():
    """Получение статуса системы отслеживания взгляда"""
    status = gaze_tracker.get_status()
    
    return StatusResponse(
        is_running=status["is_running"],
        fps=status["fps"],
        current_gaze=status["current_gaze"],
        timestamp=time.time()
    )


@app.get("/health", summary="Проверить здоровье сервиса")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    print("Запуск Gaze Tracking API...")
    print("Сервер доступен по адресу: http://127.0.0.1:8000")
    print("Документация API: http://127.0.0.1:8000/docs")
    print("Для выхода нажмите Ctrl+C")
    uvicorn.run(app, host="0.0.0.0", port=8000)