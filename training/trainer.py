"""
Класс для обучения и валидации модели
"""
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import numpy as np
from tqdm import tqdm
import os
import time

class Trainer:
    """Класс для управления процессом обучения"""
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        # Ранняя остановка
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Шедулер обучения
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, gaze_points) in enumerate(progress_bar):
            # Перемещение данных на устройство
            images = images.to(self.device)
            gaze_points = gaze_points.to(self.device)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Прямой проход
            predictions = self.model(images)
            
            # Вычисление потерь
            loss = self.criterion(predictions, gaze_points)
            
            # Обратный проход
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            
            # Вычисление MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(predictions - gaze_points)).item()
            running_mae += mae
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mae': mae
            })
        
        # Средние значения за эпоху
        epoch_loss = running_loss / len(self.train_loader)
        epoch_mae = running_mae / len(self.train_loader)
        
        return epoch_loss, epoch_mae
    
    def validate(self):
        """Валидация модели"""
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for images, gaze_points in self.val_loader:
                images = images.to(self.device)
                gaze_points = gaze_points.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, gaze_points)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(predictions - gaze_points)).item()
        
        val_loss /= len(self.val_loader)
        val_mae /= len(self.val_loader)
        
        return val_loss, val_mae
    
    def train(self):
        """Полный цикл обучения"""
        print(f"Начало обучения модели на {self.device}")
        print(f"Количество обучающих параметров: {self.model.get_num_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Обучение
            train_loss, train_mae = self.train_epoch(epoch)
            
            # Валидация
            val_loss, val_mae = self.validate()
            
            # Обновление шедулера
            self.scheduler.step(val_loss)
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # Вывод статистики
            print(f"\nЭпоха {epoch+1}/{self.config.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохранение лучшей модели
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"  ✓ Сохранена лучшая модель (loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  Early stopping counter: {self.patience_counter}/{self.config.patience}")
            
            # Проверка ранней остановки
            if self.patience_counter >= self.config.patience:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break
        
        # Общее время обучения
        total_time = time.time() - start_time
        print(f"\nОбучение завершено за {total_time:.2f} секунд")
        
        return self.history
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        if is_best:
            torch.save(checkpoint, self.config.model_save_path)
        
        # Также сохраняем последнюю модель
        last_model_path = os.path.join(
            self.config.checkpoint_dir, 
            f"gaze_model_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, last_model_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Загрузка чекпоинта модели"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Модель загружена из {checkpoint_path}")
        print(f"Эпоха: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint