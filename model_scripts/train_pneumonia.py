# Pneumonia Detection using ResNet50
from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
FEATURE_EXTRACTION_EPOCHS = 15
FINE_TUNE_EPOCHS = 8
FINE_TUNE_UNFREEZE = 30


def _set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _resolve_paths() -> tuple[Path, Path, Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "datasets" / "pneumonia"

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    for folder in (train_dir, val_dir, test_dir):
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

    model_path = project_root / "models" / "pneumonia_model.keras"
    best_path = project_root / "models" / "pneumonia_best.keras"
    metadata_path = project_root / "models" / "pneumonia_metadata.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir, test_dir, best_path, metadata_path


def _build_generators(train_dir: Path, val_dir: Path, test_dir: Path):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_generator = eval_datagen.flow_from_directory(
        str(val_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    test_generator = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def _build_model() -> tuple[Model, Model]:
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def _compile(model: Model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def _class_weights(train_generator) -> dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    return {index: float(value) for index, value in enumerate(weights)}


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)

    return best_t, best_f1


def main() -> None:
    _set_reproducibility(SEED)
    train_dir, val_dir, test_dir, best_path, metadata_path = _resolve_paths()
    model_path = metadata_path.parent / "pneumonia_model.keras"

    train_generator, val_generator, test_generator = _build_generators(train_dir, val_dir, test_dir)
    print("Class mapping:", train_generator.class_indices)

    class_weights = _class_weights(train_generator)
    print("Class weights:", class_weights)

    model, base_model = _build_model()
    _compile(model, learning_rate=1e-4)

    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint(str(best_path), monitor="val_auc", mode="max", save_best_only=True),
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FEATURE_EXTRACTION_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    for layer in base_model.layers[-FINE_TUNE_UNFREEZE:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    _compile(model, learning_rate=1e-5)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    if best_path.exists():
        model = load_model(best_path)

    val_scores = model.predict(val_generator, verbose=1).reshape(-1)
    val_true = val_generator.classes.astype(int)
    best_threshold, val_best_f1 = _best_threshold(val_true, val_scores)
    print(f"Best validation threshold: {best_threshold:.3f} (F1={val_best_f1:.4f})")

    test_metrics = model.evaluate(test_generator, verbose=1)
    metric_names = model.metrics_names
    print("Test metrics:")
    for name, value in zip(metric_names, test_metrics):
        print(f"{name}: {value:.4f}")

    test_scores = model.predict(test_generator, verbose=1).reshape(-1)
    y_true = test_generator.classes.astype(int)
    y_pred = (test_scores >= best_threshold).astype(int)

    inverse_class_indices = {value: key for key, value in test_generator.class_indices.items()}
    labels = [inverse_class_indices[index] for index in sorted(inverse_class_indices)]

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0))

    model.save(model_path)

    metadata = {
        "image_size": IMG_SIZE,
        "best_threshold": round(float(best_threshold), 4),
        "validation_best_f1": round(float(val_best_f1), 4),
        "class_indices": test_generator.class_indices,
        "seed": SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
