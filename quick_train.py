"""
Скрипт для быстрого тестирования обучения модели
"""
import torch
import torch.nn as nn
from config import Config
from models.gaze_model import GazeTrackingModel
from data.dataset_loader import DatasetLoader
from training.trainer import Trainer
import pandas as pd
import os

class QuickConfig(Config):
    """Конфигурация для быстрого обучения"""
    def __init__(self):
        super().__init__()
        self.num_epochs = 5  # Уменьшаем количество эпох
        self.train_val_split = 0.5  # Уменьшаем размер обучающей выборки
        self.batch_size = 16  # Уменьшаем размер батча для более быстрой итерации
        self.num_workers = 2  # Уменьшаем количество воркеров

def create_subset_dataset():
    """Создает подмножество датасета для быстрого обучения"""
    # Загружаем оригинальный CSV файл
    csv_path = './data/annotations.csv'
    df = pd.read_csv(csv_path)
    
    # Берем только небольшое подмножество данных
    subset_size = 2000  # Берем только 2000 примеров вместо ~100,000
    df_subset = df.sample(n=subset_size, random_state=42)
    
    # Сохраняем подмножество в новый CSV файл
    subset_csv_path = './data/annotations_subset.csv'
    df_subset.to_csv(subset_csv_path, index=False)
    
    print(f"Создан подмножество датасета с {len(df_subset)} примерами")
    print(f"Файл сохранен как {subset_csv_path}")
    
    return subset_csv_path

def quick_train():
    """Быстрое обучение модели на подмножестве данных"""
    # Создаем подмножество датасета
    subset_csv_path = create_subset_dataset()
    
    # Используем упрощенную конфигурацию
    config = QuickConfig()
    
    # Определяем устройство
    device = config.get_device()
    print(f"Используемое устройство: {device}")
    print(f"Конфигурация: {config.__dict__}")
    
    # Загружаем датасет
    print("\nЗагрузка датасета...")
    dataset_loader = DatasetLoader(config)
    
    # Загружаем датасет из подмножества
    train_loader, val_loader = dataset_loader.load_custom_dataset(
        csv_path=subset_csv_path,
        images_dir='./data'  # Корневая директория для путей к изображениям
    )
    
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    
    # Создаем модель
    print("\nСоздание модели...")
    model = GazeTrackingModel(
        backbone=config.backbone,
        num_gaze_points=config.num_gaze_points,
        dropout_rate=config.dropout_rate,
        pretrained=config.pretrained
    ).to(device)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Добавляем оптимизатор с весами для регуляризации
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Создаем тренера
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # Обучение модели
    print("\nНачало быстрого обучения...")
    history = trainer.train()
    
    print("\nБыстрое обучение завершено!")
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, config.model_save_path)
    print(f"Модель сохранена в: {config.model_save_path}")
    
    # Удаляем временный файл
    os.remove(subset_csv_path)
    print(f"Временный файл {subset_csv_path} удален")

if __name__ == "__main__":
    print("Запуск быстрого обучения модели...")
    print("После завершения обучения модель будет сохранена в checkpoints/gaze_model_best.pth")
    print("Затем можно запустить main.py, и она автоматически загрузит предварительно обученную модель")
    quick_train()