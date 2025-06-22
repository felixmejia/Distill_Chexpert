from torchvision import models
import torch.nn as nn

def get_teacher_model(num_classes=14):
    """
    Modelo Teacher: DenseNet121 pre-entrenado en ImageNet.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_student_model(num_classes=14):
    """
    Modelo Student: MobileNetV2 pre-entrenado en ImageNet. Es más ligero.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )
    return model