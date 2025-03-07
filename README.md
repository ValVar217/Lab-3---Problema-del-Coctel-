<h1 align="center"> Lab-3 - 🍸 Fiesta-de-Coctel- </h1>  

# INTRODUCCIÓN
La presente Guia/Proyecto tiene como proposito poder analizar distintas fuentes sonoras que se mezclan simulando una "fiesta de Coctel", en donde queremos extraer una señal en especifico de nuestro interes dentro de un conjunto de señales que estan superpuestas representando así el procesamiento de audio, con aplicaciones de reconocimiento de voz, la cancelacion de ruido y la mejora en la calidad del sonido que queremos procesar. Para lograr este objetivo, se emplearon herramientas computacionales que en nuestro caso es Spyder y técnicas de análisis en frecuencia en donde tenemos la Transformada de Fourier Discreta (DFT) y la Transformada Rápida de Fourier (FFT). Además de ello, se utilizaron métodos de separación de fuentes (Audios), como el Análisis de Componentes Independientes (ICA) y el Beamforming, esto con el fin de aislar la voz de uno de los tres estudiantes que hicimos parte del proyecto a partir de las señales capturadas por un conjunto de (tres) micrófonos. De esta manera, el presente informe detalla el procedimiento que se siguio en la práctica, los resultados obtenidos y su análisis de los métodos utilizados.   

## Captura de Audio:
Para esto, Tomamos las medidas correspondientes al espacio/salon en donde se ubicarian tanto las personas como tambien los tres microfonos para la toma de audio. En donde las medidas correspondientes al espacio son:   
🔵 Ancho: 181 cm  
🔵 Largo: 146 cm     

![WhatsApp Image 2025-03-07 at 10 55 13 AM](https://github.com/user-attachments/assets/0018fbec-6fe0-44b5-b9ed-eb313485d229)        
  |*Figura 1: Evidencia de medicion del salon para la toma de audio.*|  

Posteriormente, se realizo la ubicacion en lugares estrategicos para el optimo registro del audio para cada uno de los integrantes   

![WhatsApp Image 2025-03-07 at 11 00 52 AM](https://github.com/user-attachments/assets/b61721c3-caa0-4175-949e-316d22924132)    
  |*Figura 2: Ubicaciones de los Tres (3) Microfonos para la toma de audio.*|

Por ultimo, realizamos la medición de lo que sería la distancia de cada microfono hasta la persona correspondiente a cada uno (más cercana) para poder hacer un mejor analisis frente a lo que vendrian siendo nuestros resultados 

![WhatsApp Image 2025-03-07 at 11 09 01 AM](https://github.com/user-attachments/assets/feb5865a-0875-4adc-869e-386ed4e6c0fe)      
  |*Figura 3: Distancia de microfono a una fuente sonora (Persona).*|

<h1 align="center"> GUIA DE USUARIO </h1>      

## Analisis y Resultados
Para iniciar, en el encabezado de nuestro codigo podemos encontrar la inicicalizacion de las librerias correspondientes para el optimo funcionamiento del codigo para el procesamiento de señales de audio en la práctica de laboratorio, se utilizo un conjunto de librerías especializadas en análisis y manipulación de señales sonoras: 
```python
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import noisereduce as nr
 ```
Teniendo en cuenta esta primera parte de codigo, es importante que podamos identificar que funcion tiene cada una de las librerias utilizadas, por lo que tenemos que:  
▫️**os**: Nos permite el anejo de archivos y rutas.  
▫️**librosa**: Es una biblioteca utilizada para el manejo de arrays y operaciones matemáticas.
▫️**librosa.display**: Se utiliza para el procesamiento de audio.  
▫️**numpy**: Se usa para cálculos numéricos y operaciones matriciales.    
▫️**matplotlib.pyplot**: Se emplea para la generación de gráficos y visualización de datos.   
▫️**soundfile**: Facilita la lectura y escritura de archivos de audio en diferentes formatos.    
▫️**pydub**: Esta fue utilizada para la conversión y manipulación de archivos de audio en formato de MP3.    
▫️**noisereduce**: Ayuda a la reducción de ruido en las señales de audio, mejorando la calidad de los audios al eliminar sonidos no deseados.    

Posteriormente tenemos una lista llamada "rutas_audios" que almacena las rutas de acceso a los archivos de audio capturados por diferentes micrófonos.
En este caso, se tienen tres archivos en formato MP3 correspondientes a grabaciones realizadas con tres micrófonos, esto se hace con el fin de subir los debidos audios al programa y poder realizar los debidos procedimientos con ellos.
```python  
# Rutas de los archivos de audio
rutas_audios = [
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 1.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 2.mp3",
    "C:/Users/Esteban/Videos/PRUEBA/Proyecto predeterminado/MICROFONO 3.mp3"
 ```

Para la parte de codigo a continuacion lo que buscamos es la carga del audio en formato MP3, ajustando la frecuencia de muestreo a 44.1 kHz, recorta los primeros 20 segundos y normaliza la señal dividiéndola por su valor absoluto máximo y su segunda parte es con el fin de calcular la potencia de la señal, en donde usamos la **Transformada de Fourier (STFT)** para estimar el ruido de fondo al igual que la **relación señal-ruido (SNR)** en decibeles:  

```python 
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
 ```
**Analisis 👆**  
La adquisición de la señal se dio por medio de micrófonos de celular por un tiempo de 20segundos teniendo en cuenta que en todos se configuró una frecuencia de muestreo de 44.1 kHz se puede encontrar la configuración en audio.set_frame_rate(44100), en cuanto a los niveles de cuantificación es importante conocer que el audio proviene de archivos MP3, que ya están comprimidos y cuantizados con una resolución específica en el momento de cargar el audio con AudioSegment.from_file(), este está en un formato de enteros. Luego, al extraer las muestras con np.array(audio_recortado.get_array_of_samples(), dtype=np.float32), las muestras se convierten a valores en punto flotante de 32 bits permitiéndonos ajustan el rango de valores entre -1 y 1. El SNR es calculado por medio de las potencias, calcular_snr(signal), se calcula la potencia de la señal se eleva al cuadrado cada muestra de la señal y luego se saca el promedio, obteniendo la potencia media de la señal, para estimar el ruido, se aplica la STFT para obtener el espectro de frecuencias de la señal se calcula la potencia promedio del ruido elevándolo al cuadrado y sacando el promedio finalmente el SNR se calcula según su ecuación.  





```python  
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
 ```  

Para la siguiente parte, lo que se hace es Cargar los archivos de audio, encontrar la amplitud máxima entre todas las señales y el tiempo de duración máximo:  
```python    
audio_names = ["MICROFONO 1", "MICROFONO 2", "MICROFONO 3"]
plt.figure(figsize=(12, 8))
max_amplitude = 0
max_time = 0
for i, file in enumerate(rutas_audios):
    y, sr = librosa.load(file, sr=None)
    max_amplitude = max(max_amplitude, np.max(np.abs(y)))
    max_time = max(max_time, len(y) / sr) 
 ``` 

![WhatsApp Image 2025-03-07 at 3 41 08 PM](https://github.com/user-attachments/assets/32373e9b-767f-445f-bac5-735f3523ecb6)    
  |*Figura 4: Señal del Microfono 1.*|    

![WhatsApp Image 2025-03-07 at 3 41 08 PM (1)](https://github.com/user-attachments/assets/d295e268-a5a3-4bce-97dc-edd6a5423184)    
  |*Figura 5: Señal del Microfono 2.*|     

![WhatsApp Image 2025-03-07 at 3 41 08 PM (2)](https://github.com/user-attachments/assets/63da823f-e528-4469-a2da-9ac1108d23a7)  
  |*Figura 6: Señal del Microfono 3.*|   

  





