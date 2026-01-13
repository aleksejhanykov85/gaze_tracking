"""
Улучшенная архитектура нейронной сети для отслеживания взгляда
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImprovedGazeModel(nn.Module):
    """Улучшенная модель для предсказания координат взгляда"""
    
    def __init__(self, backbone='mobilenet_v2', num_gaze_points=2, dropout_rate=0.3, pretrained=True):
        super(ImprovedGazeModel, self).__init__()
        
        self.backbone_name = backbone
        self.num_gaze_points = num_gaze_points
        
        # Выбор backbone архитектуры
        if backbone == 'resnet18':
            if pretrained:
                base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet18(weights=None)
            num_features = 512
        elif backbone == 'resnet34':
            if pretrained:
                base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet34(weights=None)
            num_features = 512
        elif backbone == 'mobilenet_v2':
            if pretrained:
                base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                base_model = models.mobilenet_v2(weights=None)
            num_features = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Извлечение features из backbone
        if 'resnet' in backbone:
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif 'mobilenet' in backbone:
            self.features = base_model.features
        
        # Дополнительные слои для улучшения точности
        self.feature_enhancer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )
        
        # Регрессионная головка для предсказания взгляда
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_gaze_points),
            nn.Tanh()  # Выход в диапазоне [-1, 1] для более точного позиционирования
        )
        
        # Инициализация весов
        self._initialize_weights()
    
    def forward(self, x):
        """Прямой проход"""
        # Извлечение признаков
        features = self.features(x)
        
        # Адаптивная пуллинг в зависимости от backbone
        if 'resnet' in self.backbone_name:
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Выравнивание
        features = torch.flatten(features, 1)
        
        # Улучшение признаков
        enhanced_features = self.feature_enhancer(features)
        
        # Регрессия координат взгляда
        gaze_prediction = self.regressor(enhanced_features)
        
        return gaze_prediction
    
    def _initialize_weights(self):
        """Инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self):
        """Получение количества параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiModalGazeModel(nn.Module):
    """Улучшенная модель с дополнительными входными данными (например, положение головы)"""
    
    def __init__(self, pretrained=True):
        super(MultiModalGazeModel, self).__init__()
        
        # Визуальный поток (обработка изображения глаз)
        if pretrained:
            self.eye_net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.eye_net = models.mobilenet_v2(weights=None)
        self.eye_net.classifier = nn.Identity()  # Удаляем последний слой
        
        # Поток для положения головы (3 угла)
        self.head_pose_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Объединенный классификатор
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 128, 256),  # 1280 от eye_net + 128 от head_pose_net
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()  # Выход в диапазоне [-1, 1] для более точного позиционирования
        )
    
    def forward(self, eye_image, head_pose):
        eye_features = self.eye_net(eye_image)
        head_features = self.head_pose_net(head_pose)
        
        combined = torch.cat([eye_features, head_features], dim=1)
        gaze = self.fusion(combined)
        
        return gaze
