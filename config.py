"""
Конфигурационные параметры проекта
"""
import torch
import os

class Config:
    """Класс для управления настройками проекта"""
    
    def __init__(self):
        # Пути к данным
        self.data_dir = "./data"
        self.checkpoint_dir = "./checkpoints"
        self.model_save_path = os.path.join(self.checkpoint_dir, "gaze_model_best.pth")
        
        # Параметры датасета
        self.image_size = (224, 224)  # Размер входного изображения
        self.batch_size = 32
        self.num_workers = 4
        self.train_val_split = 0.8
        
        # Параметры модели
        self.backbone = "resnet18"  # Можно использовать: resnet18, resnet34, mobilenet_v2
        self.num_gaze_points = 2    # Предсказываем (x, y) координаты взгляда
        self.dropout_rate = 0.3
        self.pretrained = True
        
        # Параметры обучения
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.patience = 10  # Для ранней остановки
        
        # Параметры реального времени
        self.camera_id = 0  # ID камеры по умолчанию
        self.realtime_show_fps = True
        
        # Создание директорий
        self._create_directories()
    
    def _create_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_device(self):
        """Определение устройства для вычислений (GPU/CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")