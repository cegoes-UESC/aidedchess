from ultralytics import YOLO
import torch
from torch import nn

weights = [1, 0.25, 0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

model = YOLO("pose_models/final.pt")


def add_weights(trainer):

    m = trainer.model
    m.criterion = m.init_criterion()
    m.criterion.bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(weights).to(trainer.device)
    )


model.add_callback("on_train_start", add_weights)
model.train(data="chess.yaml", batch=1, epochs=1)
