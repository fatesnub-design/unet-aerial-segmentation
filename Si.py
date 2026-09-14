import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuración
PATCH_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 5
VAL_SPLIT = 0.1

# Mapeo de colores RGB a etiquetas
COLOR_MAPPING = {
    (155, 155, 155): 0,  # Unlabelled
    (132, 41, 246): 1,  # Land
    (110, 193, 228): 2,  # Road
    (254, 221, 58): 3,  # Vegetation
    (226, 169, 41): 4,  # Water
    (60, 16, 152): 5  # Building
}

# Ruta al dataset (ajustar según necesidad)
dataset_path = Path(r'C:\Users\bici_\Downloads\Semantic segmentation dataset')


def load_dataset(base_path):
    """Carga el dataset con la estructura específica de Tile X/images, Tile X/masks"""
    image_patches = []
    mask_patches = []

    # Recorrer cada carpeta Tile X
    for tile_dir in sorted(base_path.glob('Tile *')):
        if not tile_dir.is_dir():
            continue

        images_dir = tile_dir / 'images'
        masks_dir = tile_dir / 'masks'

        # Verificar estructura
        if not images_dir.exists():
            print(f"¡Advertencia! No existe {images_dir}")
            continue
        if not masks_dir.exists():
            print(f"¡Advertencia! No existe {masks_dir}")
            continue

        # Procesar cada par imagen-máscara
        for img_path in sorted(images_dir.glob('*.jpg')):
            mask_path = masks_dir / f"{img_path.stem}.png"

            if not mask_path.exists():
                print(f"¡Advertencia! No existe máscara para {img_path.name}")
                continue

            try:
                # Cargar imagen y máscara
                image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)

                # Convertir máscara RGB a etiquetas
                label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                for color, label in COLOR_MAPPING.items():
                    label_mask[(mask == color).all(axis=2)] = label

                # Dividir en parches
                for i in range(0, image.shape[0], PATCH_SIZE):
                    for j in range(0, image.shape[1], PATCH_SIZE):
                        patch_img = image[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                        patch_mask = label_mask[i:i + PATCH_SIZE, j:j + PATCH_SIZE]

                        if patch_img.shape[0] == PATCH_SIZE and patch_img.shape[1] == PATCH_SIZE:
                            image_patches.append(patch_img)
                            mask_patches.append(patch_mask)

            except Exception as e:
                print(f"Error procesando {img_path.name}: {str(e)}")

    return np.array(image_patches), np.array(mask_patches)


# Cargar datos
print("Cargando dataset...")
X, y = load_dataset(dataset_path)

if len(X) == 0:
    print("\nERROR: No se cargaron datos. Verifica:")
    print(f"1. La ruta: {dataset_path}")
    print("2. Que existan archivos .jpg en Tile */images/")
    print("3. Que existan archivos .png correspondientes en Tile */masks/")
    exit()

# Preprocesamiento
print(f"\nDatos cargados: {len(X)} parches")
X = X.astype('float32') / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(COLOR_MAPPING))

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
print(f"\nConjunto de entrenamiento: {len(X_train)} muestras")
print(f"Conjunto de prueba: {len(X_test)} muestras")


# Definición del modelo U-Net
def build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=len(COLOR_MAPPING)):
    inputs = tf.keras.Input(input_shape)

    # --- Encoder ---
    # Bloque 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bloque 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bloque 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bloque 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # --- Centro ---
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # --- Decoder ---
    # Bloque 6
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Bloque 7
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Bloque 8
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Bloque 9
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Capa de salida
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Crear el modelo
model = build_unet()
model.summary()


# Métricas personalizadas
def jaccard_index(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=len(COLOR_MAPPING))

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    return tf.reduce_mean(iou)


# Compilación
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', jaccard_index])

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5',
                    monitor='val_jaccard_index',
                    mode='max',
                    save_best_only=True,
                    verbose=1),
    EarlyStopping(monitor='val_loss',
                  patience=5,
                  restore_best_weights=True),
    CSVLogger('training_history.csv')
]

# Entrenamiento
print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluación
print("\nEvaluando modelo...")
results = model.evaluate(X_test, y_test, verbose=1)
print(f"\nResultados finales:")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Jaccard Index (IoU): {results[2]:.4f}")


# Visualización de predicciones
def plot_sample_prediction(index):
    sample_image = X_test[index]
    sample_mask = y_test[index]
    pred_mask = model.predict(np.expand_dims(sample_image, axis=0))[0]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sample_image)
    plt.title("Imagen de entrada")

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(sample_mask, axis=-1))
    plt.title("Máscara real")

    plt.subplot(1, 3, 3)
    plt.imshow(np.argmax(pred_mask, axis=-1))
    plt.title("Predicción")

    plt.show()


# Mostrar 3 ejemplos
for i in range(3):
    plot_sample_prediction(np.random.randint(0, len(X_test)))

# Guardar modelo final
model.save('unet_aerial_segmentation.h5')
print("\nModelo guardado como 'unet_aerial_segmentation.h5'")