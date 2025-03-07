import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import welch
from sklearn.decomposition import FastICA
from colorama import Fore
from pydub import AudioSegment
rutas_audios = [
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 1.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 2.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 3.mp3"
]

# --- Análisis de SNR, reducción de ruido y gráficas ---
def cargar_audio(ruta, duracion_ms=20000):
    audio = AudioSegment.from_file(ruta, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(44100)
    audio_recortado = audio[:duracion_ms]
    samples = np.array(audio_recortado.get_array_of_samples(), dtype=np.float32)
    samples /= np.max(np.abs(samples))
    return samples

def calcular_snr(signal):
    power_signal = np.mean(signal ** 2)
    stft = np.abs(librosa.stft(signal))
    noise_magnitude = np.percentile(stft, 10, axis=1)
    power_noise = np.mean(noise_magnitude ** 2)
    snr = -10 * np.log10(power_signal / power_noise)
    return snr, signal

for ruta in rutas_audios:
    if os.path.exists(ruta):
        audio = cargar_audio(ruta)
        snr, raw_signal = calcular_snr(audio)
        print(f"SNR de {os.path.basename(ruta)} (20s): {snr:.2f} dB")
        start_sample = np.where(np.abs(raw_signal) > 0.01)[0][0]
        raw_signal = raw_signal[start_sample:]
        plt.figure(figsize=(10, 4))
        plt.plot(raw_signal, color='blue')
        plt.title(f"SEÑAL RUIDO DE {os.path.basename(ruta)}")
        plt.xlabel("Tiemp(s)")
        plt.ylabel("Muestras")
        plt.grid()
        plt.show()
    else:
        print(f"ERROR: No se encontró el archivo {ruta}")

audio_names = ["MICROFONO 1", "MICROFONO 2", "MICROFONO 3"]
plt.figure(figsize=(12, 8))
max_amplitude = 0
max_time = 0
for i, file in enumerate(rutas_audios):
    y, sr = librosa.load(file, sr=None)
    max_amplitude = max(max_amplitude, np.max(np.abs(y)))
    max_time = max(max_time, len(y) / sr)


plt.tight_layout()
plt.show()
# --- Parámetros ---
distancias = [0, 0.8]  # Distancia entre micrófonos en metros
velocidad_sonido = 343  # Velocidad del sonido en m/s
rutas_audios = [
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 1.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 2.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 3.mp3"
]
audio_names = ["MICROFONO 1", "MICROFONO 2", "MICROFONO 3"]
output_dir = os.getcwd()  # Guardar archivos en el directorio actual

# Función para calcular retrasos
def calcular_retraso(distancias, velocidad, sr):
    return tuple(int(d / velocidad * sr) for d in distancias)

# Función para aplicar beamforming
def beamforming(signals, delay):
    num_mics = signals.shape[1]
    beamformed_signal = np.zeros(len(signals))
    for i, delay_i in enumerate(delay):
        beamformed_signal += np.roll(signals[:, i], delay_i)
    return beamformed_signal / num_mics

# Función para calcular SNR
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal**2) if np.mean(señal**2) > 0 else 1e-10
    potencia_ruido = np.mean(ruido**2) if np.mean(ruido**2) > 0 else 1e-10
    return 10 * np.log10(potencia_señal / potencia_ruido)

# --- Carga de audios ---
muestras_audios = []
sample_rate = None

for ruta in rutas_audios:
    if os.path.exists(ruta):
        y, sr = librosa.load(ruta, sr=None)
        muestras_audios.append(y)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError("Las tasas de muestreo de los audios no coinciden.")
    else:
        print(f"ERROR: No se encontró el archivo {ruta}")

# Asegurar que todas las señales tengan la misma longitud
longitud_max = max(len(y) for y in muestras_audios)
muestras_audios = [np.pad(y, (0, longitud_max - len(y))) for y in muestras_audios]

# Convertir a array y calcular retraso
audio_mix = np.vstack(muestras_audios).T
retraso = calcular_retraso(distancias, velocidad_sonido, sample_rate)

# Aplicar Beamforming
beamformed_signal = beamforming(audio_mix, retraso)
beamformed_signal_denoised = nr.reduce_noise(y=beamformed_signal, sr=sample_rate, stationary=True)

# Guardar señal procesada
output_file_beamformed = os.path.join(output_dir, "señal_beamformed.wav")
sf.write(output_file_beamformed, beamformed_signal_denoised, sample_rate)
print(f"Señal beamformed guardada en: {output_file_beamformed}")

# Aplicar ICA
ica = FastICA(n_components=2)
señales_separadas = ica.fit_transform(audio_mix)
señal_ica = señales_separadas[:, 0]

# Reducir la frecuencia de muestreo de la señal ICA
sample_rate_reducido = sample_rate // 2
señal_ica_reducida = librosa.resample(señal_ica, orig_sr=sample_rate, target_sr=sample_rate_reducido)

output_file_ica = os.path.join(output_dir, "señal_ica.wav")
sf.write(output_file_ica, señal_ica_reducida, sample_rate_reducido)
print(f"Señal ICA guardada en: {output_file_ica}")

# Cálculo de SNR
ruido_estimado = audio_mix[:, 1] - audio_mix[:, 0]
snr_beam = calcular_snr(beamformed_signal_denoised, ruido_estimado)
snr_ica = calcular_snr(señal_ica, ruido_estimado)
print(Fore.BLUE + f"SNR después de Beamforming: {snr_beam:.2f} dB")
print(Fore.BLUE + f"SNR después de ICA: {snr_ica:.2f} dB")

# --- Graficación de señales individuales ---
for i, y in enumerate(muestras_audios):
    plt.figure(figsize=(12, 10))

    # --- Forma de onda ---
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sample_rate)
    plt.title(f"Forma de onda - {audio_names[i]}", fontsize=14)
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("Amplitud (mV)", fontsize=12)

    # --- Espectro de frecuencia ---
    N = len(y)
    T = 1.0 / sample_rate
    fft_values = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, T)[:N // 2]
    fft_values = np.abs(fft_values[:N // 2])

    plt.subplot(3, 1, 2)
    plt.semilogx(freqs, fft_values, color='r')
    plt.title(f"Espectro de Frecuencia - {audio_names[i]}", fontsize=14)
    plt.xlabel("Frecuencia (Hz)", fontsize=12)
    plt.ylabel("Magnitud", fontsize=12)

    # --- Densidad Espectral de Potencia (PSD) ---
    freqs, psd = welch(y, fs=sample_rate, nperseg=1024)

    plt.subplot(3, 1, 3)
    plt.semilogx(freqs, psd, color='g')
    plt.title(f"Densidad Espectral de Potencia - {audio_names[i]}", fontsize=14)
    plt.xlabel("Frecuencia (Hz)", fontsize=12)
    plt.ylabel("PSD (log)", fontsize=12)
    plt.yscale("log")  # Escala logarítmica en el eje Y

    plt.tight_layout(pad=3.0)
    plt.show()

# Graficar comparación de señales procesadas
plt.figure(figsize=(12, 6))
plt.plot(beamformed_signal_denoised, label="Señal Beamformed", alpha=0.7)
plt.plot(señal_ica_reducida, label="Señal ICA (Frecuencia Reducida)", alpha=0.7)
plt.title("Comparación de Señales Procesadas")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.show()


