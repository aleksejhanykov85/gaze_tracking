"""
Загрузка и подготовка датасетов для отслеживания взгляда
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

class GazeDataset(Dataset):
    """Кастомный датасет для отслеживания взгляда"""
    
    def __init__(self, csv_file, root_dir, transform=None, is_train=True):
        """
        Args:
            csv_file (string): Путь к CSV файлу с аннотациями
            root_dir (string): Директория с изображениями
            transform (callable, optional): Трансформации для изображений
            is_train (bool): Флаг для обучения/тестирования
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform or self._get_default_transform(is_train)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Загрузка изображения
        img_path = self.annotations.iloc[idx, 0]
        
        # Если путь начинается с 'images/', добавляем к root_dir 'MPIIGaze'
        if img_path.startswith('images/'):
            img_name = os.path.join(self.root_dir, 'MPIIGaze', img_path)
        else:
            img_name = os.path.join(self.root_dir, img_path)
            
        image = Image.open(img_name).convert('RGB')
        
        # Загрузка меток (координаты взгляда на экране)
        # Используем screen_gaze_x и screen_gaze_y вместо нормализованных координат
        screen_gaze_x = self.annotations.iloc[idx, 7]  # screen_gaze_x находится в 8-й колонке (индекс 7)
        screen_gaze_y = self.annotations.iloc[idx, 8]  # screen_gaze_y находится в 9-й колонке (индекс 8)
        # Нормализуем координаты экрана к диапазону [0, 1]
        # Предполагаем максимальное разрешение 1920x1080, но можем нормализовать по максимальным значениям в датасете
        max_screen_x = 1920.0  # Максимальное значение по X
        max_screen_y = 1080.0  # Максимальное значение по Y
        gaze_x = screen_gaze_x / max_screen_x
        gaze_y = screen_gaze_y / max_screen_y
        gaze_point = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        return image, gaze_point
    
    def _get_default_transform(self, is_train):
        """Стандартные трансформации для изображений"""
        if is_train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

class DatasetLoader:
    """Класс для загрузки и управления датасетами"""
    
    def __init__(self, config):
        self.config = config
    
    def load_gazecapture(self, dataset_path=None):
        """
        Загрузка датасета GazeCapture (пример структуры)
        Требует предварительного скачивания датасета
        """
        dataset_path = dataset_path or self.config.data_dir
        
        # Проверяем, есть ли файл annotations.csv в корне dataset_path
        csv_file = os.path.join(dataset_path, "annotations.csv")
        
        # Создание полного датасета
        full_dataset = GazeDataset(
            csv_file=csv_file,
            root_dir=dataset_path,  # root_dir - корневая директория, из которой будут строиться пути к изображениям
            is_train=True
        )
        
        # Разделение на train/validation
        train_size = int(self.config.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Создание DataLoader'ов
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
    
    def load_custom_dataset(self, csv_path, images_dir):
        """Загрузка кастомного датасета с разделением на train/validation"""
        full_dataset = GazeDataset(
            csv_file=csv_path,
            root_dir=images_dir,
            is_train=True
        )
        
        # Разделение на train/validation
        train_size = int(self.config.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Создание DataLoader'ов
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
    
    def _create_sample_annotation(self, dataset_path):
        """Создание примера файла аннотаций (для демонстрации)"""
        import random
        
        # Пример создания CSV файла
        data = []
        for i in range(100):  # 100 примеров для демонстрации
            data.append({
                'image_path': f'image_{i:04d}.jpg',
                'gaze_x': random.uniform(0, 1),
                'gaze_y': random.uniform(0, 1)
            })
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(dataset_path, "annotations.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Создан пример аннотационного файла: {csv_path}")
        print("Для реального использования замените на свои данные")