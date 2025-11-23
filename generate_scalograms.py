import os
import librosa
import pywt
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Caminhos das pastas
DATA_DIR = "data"
OUTPUT_DIR = "scalograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scalogram(audio_path, output_path):
    # Carrega o áudio
    y, sr = librosa.load(audio_path, sr=None)
    # Garante duração mínima
    if len(y) < sr * 2:
        y = np.pad(y, (0, sr * 2 - len(y)))
    else:
        y = y[:sr * 2]

    # CWT com Wavelet Complex Morlet
    scales = np.arange(1, 128)
    coefficients, _ = pywt.cwt(y, scales, 'cmor1.5-1.0')
    power = np.abs(coefficients)

    # Normaliza
    power = (power - np.min(power)) / (np.max(power) - np.min(power))

    # Gera e salva o escalograma como imagem
    plt.figure(figsize=(4, 4))
    plt.imshow(power, extent=[0, 2, 1, 128], cmap='viridis', aspect='auto')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_dataset():
    for label in ['real', 'fake']:
        input_dir = os.path.join(DATA_DIR, label)
        output_subdir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_subdir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_subdir, filename.replace('.wav', '.png'))
                print(f"Gerando escalograma: {output_path}")
                generate_scalogram(audio_path, output_path)

if __name__ == "__main__":
    process_dataset()
