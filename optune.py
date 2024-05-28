import optuna
from ultralytics import YOLO


def objective(trial):
    if trial.number == 0:
        args = {"epochs": 100, "batch": 16}
    else:
        args = {
            "optimizer": "AdamW",
            "lr0": trial.suggest_float("lr0", 0.001, 0.5),
            "momentum": trial.suggest_float("momentum", 0.9, 0.999),
            "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.001),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 3, 10),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.01, 0.99),
            "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 0.1, 0.9),
            "box": trial.suggest_float("box", 0.5, 2.0),
            "cls": trial.suggest_float("cls", 6.0, 12.0),
            "pose": trial.suggest_float("pose", 12.0, 24.0),
            "kobj": trial.suggest_float("kobj", 2.0, 12.0),
            "batch": trial.suggest_int("batch", 8, 16),
            "epochs": trial.suggest_int("epochs", 25, 100),
            "dropout": trial.suggest_float("dropout", 0.05, 0.2),
        }

    model: YOLO = YOLO("models/pose_final.pt")

    model.train(
        data="chess.yaml",
        imgsz=640,
        val=False,
        fliplr=0,
        flipud=0,
        scale=0,
        translate=0,
        hsv_v=0,
        hsv_s=0,
        hsv_h=0,
        erasing=0,
        augment=False,
        **args
    )
    metrics = model.val(data="chess.yaml")

    return metrics.fitness


storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage("./optuna.log"),
)

study = optuna.create_study(
    direction="maximize",
    study_name="chess-optune",
    storage=storage,
    load_if_exists=True,
)

study.optimize(objective, n_trials=1)
