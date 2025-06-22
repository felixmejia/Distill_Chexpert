import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Función de pérdida para Knowledge Distillation.
    Combina la pérdida dura (con las etiquetas reales) y la pérdida suave (con los logits del teacher).
    """
    def __init__(self, base_criterion, teacher_model, alpha, temperature):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        # La pérdida de destilación es la divergencia KL
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, inputs, outputs, labels):
        # Pérdida estándar (Hard Loss)
        base_loss = self.base_criterion(outputs, labels)

        # Obtener logits del teacher (no se necesita gradiente para el teacher)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # Calcular pérdida de destilación (Soft Loss)
        soft_targets = F.log_softmax(outputs / self.temperature, dim=1)
        soft_labels = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        distillation_loss = self.distillation_criterion(soft_targets, soft_labels) * (self.temperature ** 2)

        # Combinar las pérdidas
        loss = self.alpha * base_loss + (1 - self.alpha) * distillation_loss
        return loss