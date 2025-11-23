import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import cv2
from tensorflow.keras.models import load_model

# Função para gerar escalograma temporário
def generate_scalogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    coef, freqs = pywt.cwt(y, scales=np.arange(1, 129), wavelet='morl')
    plt.figure(figsize=(4, 4))
    plt.imshow(np.abs(coef), extent=[0, len(y)/sr, 1, 128], cmap='viridis', aspect='auto')
    plt.axis('off')
    temp_path = "temp_scalogram.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return temp_path

# Função de previsão
def predict_audio(audio_path, model_path='audio_fake_detector_cnn.h5'):
    print(f"🔍 Analisando: {audio_path}")
    model = load_model(model_path)

    # Gera o escalograma temporário
    temp_img = generate_scalogram(audio_path)
    
    # Pré-processa a imagem
    img = cv2.imread(temp_img)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Faz a previsão
    pred = model.predict(img)[0][0]
    os.remove(temp_img)  # apaga o arquivo temporário

    if pred > 0.5:
        print(f"✅ Resultado: REAL (confiança: {pred:.2f})")
    else:
        print(f"⚠️ Resultado: FAKE (confiança: {1 - pred:.2f})")

# Teste com um áudio
if __name__ == "__main__":
    audio_teste = input("Digite o caminho do áudio que deseja testar (.wav): ")
    predict_audio(audio_teste)
