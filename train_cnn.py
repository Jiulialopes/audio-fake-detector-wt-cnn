import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Caminhos das pastas
base_dir = 'scalograms'
train_dir = base_dir  # já contém /real e /fake

# Gerador de imagens (pré-processamento + aumento de dados)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% treino / 20% validação
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compila o modelo
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Salva o modelo
model.save('audio_fake_detector_cnn.h5')
print("✅ Modelo treinado e salvo como 'audio_fake_detector_cnn.h5'")
