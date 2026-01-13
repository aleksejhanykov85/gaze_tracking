"""
Главный скрипт для обучения модели отслеживания взгляда
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

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение модели отслеживания взгляда')
    
    parser.add_argument('--dataset', type=str, default='gazecapture',
                       help='Датасет для обучения (gazecapture/custom)')
    parser.add_argument('--dataset_path', type=str, default='./data',
                       help='Путь к датасету')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Скорость обучения')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       help='Архитектура backbone (resnet18/resnet34/mobilenet_v2)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Путь к чекпоинту для продолжения обучения')
    
    return parser.parse_args()

def main():
    """Основная функция обучения"""
    # Парсинг аргументов
    args = parse_arguments()
    
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
    
    # Определение устройства
    device = config.get_device()
    print(f"Используемое устройство: {device}")
    print(f"Конфигурация: {config.__dict__}")
    
    # Загрузка датасета
    print("\nЗагрузка датасета...")
    dataset_loader = DatasetLoader(config)
    
    if args.dataset.lower() == 'gazecapture':
        train_loader, val_loader = dataset_loader.load_gazecapture(args.dataset_path)
    else:
        print("Используется демонстрационный режим с синтетическими данными")
        train_loader, val_loader = dataset_loader.load_gazecapture()
    
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    
    # Создание модели
    print("\nСоздание модели...")
    model = GazeTrackingModel(
        backbone=config.backbone,
        num_gaze_points=config.num_gaze_points,
        dropout_rate=config.dropout_rate,
        pretrained=config.pretrained
    ).to(device)
    
    # Загрузка чекпоинта если указано
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {args.resume}")
    else:
        # Проверяем, есть ли уже обученная модель в checkpoints
        if os.path.exists(config.model_save_path):
            print(f"Найдена предварительно обученная модель: {config.model_save_path}")
            print("Загрузка модели...")
            try:
                checkpoint = torch.load(config.model_save_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("Предварительно обученная модель загружена")
            except Exception as e:
                print(f"Не удалось загрузить предварительно обученную модель: {e}")
                print("Будет использована новая модель")
    
    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()  # MSE для регрессии координат
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
    # print("\nНачало обучения...")
    # history = trainer.train()
    
    # # Визуализация результатов
    # visualizer = Visualizer()
    # visualizer.plot_training_history(history)
    
    # print("\nОбучение завершено!")
    # print(f"Лучшая модель сохранена в: {config.model_save_path}")

if __name__ == "__main__":
    main()