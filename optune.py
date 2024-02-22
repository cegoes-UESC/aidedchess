import optuna
from ultralytics import YOLO
from ultralytics.utils.metrics import PoseMetrics


def objective(trial):
    args = {
        "lr0": trial.suggest_float("lr0", 0.001, 0.5),
        "momentum": trial.suggest_float("momentum", 0.1, 0.9),
        "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.001),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 1, 5),
        "warmup_momentum": trial.suggest_float("warmup_momentum", 0.01, 0.99),
        "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 0.1, 0.9),
        "box": trial.suggest_float("box", 1.0, 9.0),
        "cls": trial.suggest_float("cls", 0.5, 10.0),
        "pose": trial.suggest_float("pose", 6.0, 24.0),
        "kobj": trial.suggest_float("kobj", 1.0, 10.0),
    }

    model: YOLO = YOLO("pose_30_01_2024.pt")

    model.train(data="chess.yaml", epochs=100, imgsz=640, **args)
    validation: PoseMetrics = model.val(data="chess.yaml")

    return validation.fitness, validation.pose.mp, validation.pose.mr


study = optuna.create_study(directions=["maximize", "maximize", "maximize"])

study.optimize(objective, n_trials=10)


print(f"Study completed for: {len(study.trials)}")


trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
