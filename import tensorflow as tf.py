import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define o caminho para o conjunto de dados
data_dir = "caminho/para/os/dados"

# Define as classes do conjunto de dados (danificado e não danificado)
classes = ["danificado", "nao_danificado"]

# Define o tamanho das imagens a serem usadas no treinamento do modelo
IMG_SIZE = 224

# Carrega as imagens do conjunto de dados e as divide em um conjunto de treinamento e um conjunto de teste
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32
)

# Pré-processa as imagens do conjunto de dados
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# Define o modelo de aprendizado de máquina utilizando a arquitetura ResNet50 pré-treinada
base_model = tf.keras.applications.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Congela as camadas do modelo pré-treinado para evitar que sejam treinadas novamente
for layer in base_model.layers:
    layer.trainable = False

# Adiciona camadas personalizadas ao modelo
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(1)(global_average_layer)

# Define o modelo completo
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

# Compila o modelo
model.compile(
    optimizer=tf
