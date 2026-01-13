"""
Визуализация результатов обучения и работы модели
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict

class Visualizer:
    """Класс для визуализации различных аспектов работы модели"""
    
    @staticmethod
    def plot_training_history(history: Dict):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # График MAE
        axes[1].plot(history['train_mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_gaze_predictions(images, true_gaze, pred_gaze, num_samples=5):
        """Визуализация предсказаний взгляда"""
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(images))):
            # Денормализация изображения
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Истинное и предсказанное направление взгляда
            true = true_gaze[i].cpu().numpy()
            pred = pred_gaze[i].cpu().numpy()
            
            # Первое изображение - оригинал с векторами
            axes[i, 0].imshow(img)
            axes[i, 0].arrow(112, 112, 
                           true[0] * 50 - 50, true[1] * 50 - 50,
                           color='green', width=2, head_width=5)
            axes[i, 0].arrow(112, 112,
                           pred[0] * 50 - 50, pred[1] * 50 - 50,
                           color='red', width=2, head_width=5, alpha=0.7)
            axes[i, 0].set_title(f"True: {true}, Pred: {pred}")
            axes[i, 0].axis('off')
            
            # Второе изображение - тепловая карта ошибки
            error = np.abs(true - pred)
            axes[i, 1].imshow(np.ones_like(img) * 0.5)
            axes[i, 1].scatter([112], [112], c=[np.mean(error)], 
                             cmap='Reds', s=200, vmin=0, vmax=0.5)
            axes[i, 1].set_title(f"Error: {np.mean(error):.3f}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('gaze_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_gaze_heatmap(gaze_points, frame_shape, gaze_intensity=255):
        """Создание тепловой карты скоплений взглядов"""
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for point in gaze_points:
            x, y = int(point[0] * frame_shape[1]), int(point[1] * frame_shape[0])
            
            # Создание 2D Гауссовского распределения
            for i in range(max(0, x-10), min(frame_shape[1], x+10)):
                for j in range(max(0, y-10), min(frame_shape[0], y+10)):
                    dist = np.sqrt((i-x)**2 + (j-y)**2)
                    if dist < 10:
                        heatmap[j, i] += np.exp(-dist**2 / (2 * 5**2)) * gaze_intensity
        
        # Нормализация
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        
        # Применение цветовой карты
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored