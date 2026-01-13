"""
Главный скрипт для обучения модели отслеживания взгляда (исправленная версия)
"""
import torch
import torch.nn as nn
from config import Config
from models.gaze_model import GazeTrackingModel
from data.dataset_loader import DatasetLoader
from training.trainer import Trainer
from utils.visualizer import Visualizer
import argparse
import os

# Устанавливаем локальный кеш для torch
os.environ['TORCH_HOME'] = './torch_cache'
os.makedirs('./torch_cache', exist_ok=True)

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение модели отслеживания взгляда')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Датасет для обучения (synthetic/mpiigaze/custom)')
    parser.add_argument('--dataset_path', type=str, default='./data',
                       help='Путь к датасету')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=8,  # Уменьшен для CPU
                       help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Скорость обучения')
    parser.add_argument('--backbone', type=str, default='simple',
                       help='Архитектура backbone (simple/resnet18/resnet34)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Использовать предобученные веса')
    parser.add_argument('--resume', type=str, default=None,
                       help='Путь к чекпоинту для продолжения обучения')
    
    return parser.parse_args()

def create_simple_model():
    """Создание простой модели для быстрого тестирования"""
    class SimpleGazeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x
            
        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    return SimpleGazeModel()

def create_synthetic_dataset():
    """Создание синтетического датасета для тестирования"""
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    
    class SyntheticGazeDataset(Dataset):
        def __init__(self, num_samples=100, image_size=(224, 224)):
            self.num_samples = num_samples
            self.image_size = image_size
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Создаем синтетическое изображение
            image = torch.randn(3, *self.image_size)
            
            # Создаем синтетические координаты взгляда
            # Имитируем нормальное распределение вокруг центра
            center = torch.tensor([0.5, 0.5])
            noise = torch.randn(2) * 0.1
            gaze = torch.clamp(center + noise, 0, 1)
            
            return image, gaze
    
    # Создаем датасет
    full_dataset = SyntheticGazeDataset(num_samples=120)
    
    # Разделяем на train/val
    from torch.utils.data import random_split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def main():
    """Основная функция обучения"""
    # Парсинг аргументов
    args = parse_arguments()
    
    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ОТСЛЕЖИВАНИЯ ВЗГЛЯДА")
    print("=" * 60)
    
    # Инициализация конфигурации
    config = Config()
    
    # Обновление конфигурации из аргументов
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.backbone:
        config.backbone = args.backbone
    if args.pretrained:
        config.pretrained = True
    else:
        config.pretrained = False  # Не используем предобученные веса
    
    # Определение устройства
    device = torch.device('cpu')
    print(f"Используемое устройство: {device}")
    
    # Загрузка датасета
    print(f"\nЗагрузка датасета: {args.dataset}")
    
    if args.dataset.lower() == 'synthetic':
        print("Используются синтетические данные для тестирования")
        train_loader, val_loader = create_synthetic_dataset()
    elif args.dataset.lower() == 'mpiigaze':
        # Загрузка MPIIGaze
        dataset_loader = DatasetLoader(config)
        csv_path = os.path.join(args.dataset_path, 'MPIIGaze', 'annotations.csv')
        images_dir = os.path.join(args.dataset_path, 'MPIIGaze', 'images')
        
        if os.path.exists(csv_path):
            print(f"Загрузка MPIIGaze из {csv_path}")
            train_loader = dataset_loader.load_custom_dataset(csv_path, images_dir)
            val_loader = dataset_loader.load_custom_dataset(csv_path, images_dir)
        else:
            print(f"Файл {csv_path} не найден. Используются синтетические данные.")
            train_loader, val_loader = create_synthetic_dataset()
    else:
        print("Используются синтетические данные")
        train_loader, val_loader = create_synthetic_dataset()
    
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    print(f"Количество батчей (train): {len(train_loader)}")
    print(f"Количество батчей (val): {len(val_loader)}")
    
    # Создание модели
    print(f"\nСоздание модели с backbone: {args.backbone}")
    
    if args.backbone == 'simple':
        model = create_simple_model().to(device)
    else:
        try:
            model = GazeTrackingModel(
                backbone=config.backbone,
                num_gaze_points=config.num_gaze_points,
                dropout_rate=config.dropout_rate,
                pretrained=config.pretrained  # False для избежания загрузки
            ).to(device)
        except Exception as e:
            print(f"Ошибка при создании модели: {e}")
            print("Используется простая модель...")
            model = create_simple_model().to(device)
    
    # Загрузка чекпоинта если указано
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {args.resume}")
    
    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Создание тренера
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
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    history = trainer.train()
    
    # Визуализация результатов
    print("\nСоздание графиков обучения...")
    visualizer = Visualizer()
    visualizer.plot_training_history(history)
    
    # Тестирование модели
    print("\nТестирование обученной модели...")
    model.eval()
    with torch.no_grad():
        test_images, test_gaze = next(iter(val_loader))
        test_images = test_images.to(device)
        predictions = model(test_images)
        
        mse = criterion(predictions, test_gaze)
        mae = torch.mean(torch.abs(predictions - test_gaze))
        
        print(f"Test MSE: {mse.item():.4f}")
        print(f"Test MAE: {mae.item():.4f}")
        
        # Пример предсказания
        print("\nПримеры предсказаний:")
        for i in range(min(3, len(predictions))):
            print(f"  Изображение {i+1}:")
            print(f"    Истинное: {test_gaze[i].numpy()}")
            print(f"    Предсказанное: {predictions[i].cpu().numpy()}")
            print(f"    Ошибка: {torch.abs(predictions[i] - test_gaze[i]).numpy()}")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Лучшая модель сохранена в: {config.model_save_path}")
    print("=" * 60)
    
    # Сохранение полной истории
    history_path = os.path.join(config.checkpoint_dir, 'training_history.pt')
    torch.save({
        'history': history,
        'config': config.__dict__,
        'model_summary': str(model)
    }, history_path)
    print(f"История обучения сохранена в: {history_path}")

if __name__ == "__main__":
    main()