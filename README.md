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
▫️**os** → Nos permite el anejo de archivos y rutas.  
▫️**librosa** → Es Para el procesamiento del Audio 
▫️**librosa.display** → Se utiliza para el procesamiento de audio.  
▫️**numpy (np)** → Se usa para el manejo de arrays y operaciones matemáticas.    
▫️**matplotlib.pyplot** → Se emplea para la generación de gráficos y visualización de datos.   
▫️**soundfile** → Facilita la lectura y escritura de archivos de audio.
▫️**pydub** → Esta fue utilizada para la conversión y manipulación de archivos de audio en formato de MP3.    
▫️**noisereduce** → Ayuda a la reducción de ruido en las señales de audio, mejorando la calidad de los audios al eliminar sonidos no deseados.    

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

Complementando lo anterior, sobre los archivos de audio, se calcula el SNR y grafica la señal de audio sin ruido:  

**Calculo del SNR**  
Hay que recordar que la SNR es una métrica que cuantifica la calidad de una señal en presencia de ruido. Se define como la relación entre la potencia de la señal útil y la potencia del ruido, expresada en decibeles (dB):   

 ![image](https://github.com/user-attachments/assets/ffad5937-9c24-4e59-9ea3-dce5cb31ee35)   

```python  
for ruta in rutas_audios:
    if os.path.exists(ruta):
        audio = cargar_audio(ruta)
        snr, raw_signal = calcular_snr(audio)
        print(f"SNR de {os.path.basename(ruta)} (20s): {snr:.2f} dB")
        start_sample = np.where(np.abs(raw_signal) > 0.01)[0][0]
        raw_signal = raw_signal[start_sample:]
 ``` 
Tambien, genara los caracteristicos que queremos para la visualizacion de nuestras señales correspondientes a cada uno de los audios.
```python 
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
En este punto la señal recortada se grafica en el dominio del tiempo utilizando matplotlib. Se establece el color azul y se añaden las etiquetas correspondientes a los ejes para facilitar la interpretación de la gráfica:  
![WhatsApp Image 2025-03-07 at 3 41 08 PM](https://github.com/user-attachments/assets/32373e9b-767f-445f-bac5-735f3523ecb6)      
  |*Figura 4: Señal del Microfono 1.*|      
     _________________________________________________________________________________________________
![WhatsApp Image 2025-03-07 at 3 41 08 PM (1)](https://github.com/user-attachments/assets/d295e268-a5a3-4bce-97dc-edd6a5423184)      
  |*Figura 5: Señal del Microfono 2.*|         
     _________________________________________________________________________________________________
![WhatsApp Image 2025-03-07 at 3 41 08 PM (2)](https://github.com/user-attachments/assets/63da823f-e528-4469-a2da-9ac1108d23a7)    
  |*Figura 6: Señal del Microfono 3.*|     
     _________________________________________________________________________________________________  

Para llevar a cabo el análisis se realizó en dominio del tiempo generando gráficas del tiempo con respecto a voltaje de cada señal, y en el dominio de la frecuencia aplicando la transformada rápida de Fourier graficando de igual forma la frecuencia con respecto a su magnitud  
Las señales de audio fueron analizadas de forma independiente con el encontrar información como la potencia que contiene cada una de ellas   
Se realizó un compilación de los tres audios y se calcula la señal beamformed esperando aislar la voz de interés.
___________________________________
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

Lo que viene en lo siguiente, es con el proposito de ajustar el diseño y muestra las gráficas, definir parámetros físicos clave para análisis de señales, especificae las rutas de los archivos de audio de los micrófonos, a su vez asignar etiquetas para los micrófonos, al igual que es para poder establece el directorio de salida para guardar resultados:  
```python    
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
 ```

Ahora, lo siguiente define una función para calcular los retrasos de tiempo entre micrófonos debido a la diferencia en la distancia que recorre el sonido hasta cada micrófono  
```python   
# Función para calcular retrasos
def calcular_retraso(distancias, velocidad, sr):
    return tuple(int(d / velocidad * sr) for d in distancias)
 ```
Ahora bien, tenemos la aplicacion de Beamforming a continuacion, pues esta es como un  tipo de tectica que nos permite combinar las señales de varios microfonos para poder mejorar la captacion de la señal deseada o de nuestro interes y reducir el ruido:  
```python
# Función para aplicar beamforming
def beamforming(signals, delay):
    num_mics = signals.shape[1]
    beamformed_signal = np.zeros(len(signals))
    for i, delay_i in enumerate(delay):
        beamformed_signal += np.roll(signals[:, i], delay_i)
    return beamformed_signal / num_mics
 ```
Hay que tener presente que para ello la función beamforming aplica un retraso a cada señal y las combina para reforzar las componentes comunes y cancelar el ruido no correlacionado.

```python
# Función para calcular SNR
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal**2) if np.mean(señal**2) > 0 else 1e-10
    potencia_ruido = np.mean(ruido**2) if np.mean(ruido**2) > 0 else 1e-10
    return 10 * np.log10(potencia_señal / potencia_ruido)  
 ```


```python
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
 ```


```python
# Asegurar que todas las señales tengan la misma longitud
longitud_max = max(len(y) for y in muestras_audios)
muestras_audios = [np.pad(y, (0, longitud_max - len(y))) for y in muestras_audios]
 ```


```python
# Convertir a array y calcular retraso
audio_mix = np.vstack(muestras_audios).T
retraso = calcular_retraso(distancias, velocidad_sonido, sample_rate)
 ```


```python
# Aplicar Beamforming
beamformed_signal = beamforming(audio_mix, retraso)
beamformed_signal_denoised = nr.reduce_noise(y=beamformed_signal, sr=sample_rate, stationary=True)
 ```



```python
# Guardar señal procesada
output_file_beamformed = os.path.join(output_dir, "señal_beamformed.wav")
sf.write(output_file_beamformed, beamformed_signal_denoised, sample_rate)
print(f"Señal beamformed guardada en: {output_file_beamformed}") 
 ```


```python
# Aplicar ICA
ica = FastICA(n_components=2)
señales_separadas = ica.fit_transform(audio_mix)
señal_ica = señales_separadas[:, 0]
 ```


```python
# Reducir la frecuencia de muestreo de la señal ICA
sample_rate_reducido = sample_rate // 2
señal_ica_reducida = librosa.resample(señal_ica, orig_sr=sample_rate, target_sr=sample_rate_reducido)

output_file_ica = os.path.join(output_dir, "señal_ica.wav")
sf.write(output_file_ica, señal_ica_reducida, sample_rate_reducido)
print(f"Señal ICA guardada en: {output_file_ica}")
 ```


```python
# Cálculo de SNR
ruido_estimado = audio_mix[:, 1] - audio_mix[:, 0]
snr_beam = calcular_snr(beamformed_signal_denoised, ruido_estimado)
snr_ica = calcular_snr(señal_ica, ruido_estimado)
print(Fore.BLUE + f"SNR después de Beamforming: {snr_beam:.2f} dB")
print(Fore.BLUE + f"SNR después de ICA: {snr_ica:.2f} dB")
 ```


```python
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
 ```
Teniendo como resultado las siguientes graficas:     

![WhatsApp Image 2025-03-07 at 3 41 09 PM (3)](https://github.com/user-attachments/assets/9e3b8751-a836-4dea-bc36-3e8b347c07cb)    
  |*Figura 7: La forma de onda, el espectro de frecuencia y (PSD) de la Señal del Microfono 1.*|   
     _________________________________________________________________________________________________  
![WhatsApp Image 2025-03-07 at 3 41 09 PM (4)](https://github.com/user-attachments/assets/1ef4a4ce-3394-422e-a98e-d50f058965d5)      
  |*Figura 8: La forma de onda, el espectro de frecuencia y (PSD) de la Señal del Microfono 2.*|   
     _________________________________________________________________________________________________  
![WhatsApp Image 2025-03-07 at 3 41 09 PM (5)](https://github.com/user-attachments/assets/3b94d994-ca22-4b06-abda-2373fd886319)        
  |*Figura 9: La forma de onda, el espectro de frecuencia y (PSD) de la Señal del Microfono 3.*|   
     _________________________________________________________________________________________________

```python
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
 ```

# RESULTADOS  
Se evidencia inicialmente la carga de los audios con sus respectivas gráficas en donde podemos observar el tiempo de sonido del vacío y el tiempo en que hay una voz se grafica independiente el sonido vacío de 20 segundos, de esta grafica podemos hallar la potencia de ruido valor para calcular el SNR de cada señal del micrófono 1 obtuvimos un SNR de 14.57 dB, micrófono 2 SNR de 8.21dB y el micrófono 3 el SNR es de 10.96dB
Estos resultados son comparados con bibliografía ya que el SNR que se espera obtener de un micrófono de celular oscila entre 10 a 20 dB, el resultado del micrófono 2 es más bajo de lo normal y asociamos esto a daños del micrófono o desactualización debido que la toma fue de un celular un poco más viejo.  

Se prepresenta en la segunda gráfica colo rojo  el Espectro de Frecuencia, que muestra la Transformada de Fourier (FFT) de la señal en una escala lineal.realizamos el análisis para cada micrófono: Micrófono 1 encontramos que la máxima magnitud que presenta es alrededor de 2000 esto debido a la que la voz de la persona sexo femenino es un poco más tenue y la ubicación de distancia que tenía del micrófono, el micrófono 2 presenta una magnitud máxima cercana a 5000 pero está magnitud se mantiene medianamente constante con respecto a las otras señales asociamos esto a qué la persona de sexo masculino cuenta con un tono de voz mucho más fuerte con respecto a los otros y mantuvo contante su tono de voz, también su distancia con respecto al micrófono, micrófono 3 presenta una magnitud máxima de aproximadamente 2500, presenta un pico mucho más alto pero no duradero debido a que fue un golpe momentáneo que se le realizó al micrófono, luego de esto mantiene su magnitud entre 1000 y 2500, realizamos la comparación con el micrófono 1 e identificamos la importancia de la distancia y ubicación de la persona con respecto al micrófono ya que en los dos son personas de sexo femenino pero el micrófono 3 la persona estaba más cerca de este.   

Con respecto al eje X Frecuencia en Hz, escala lineal, representa las frecuencias presentes en la señal. La escala es logarítmica, lo que permite visualizar mejor un amplio rango de frecuencia analizando una a una las señales encontramos que Micrófono 1 el espectro muestra picos concentrados en una banda alrededor de los 100 Hz-1 kHz , esto indica que la señal captada tiene componentes en ese rango de frecuencias, posiblemente un sonido, no se observan muchas frecuencias altas, lo que sugiere que la señal es más limpia o que el micrófono tiene menor sensibilidad en frecuencias más altas.  

El micrófono 2 indica presencia de picos en el rango de 100 Hz a 1 kHz nos brinda información de que el sonido captado tiene componentes dominantes en esta banda, este rango es típico de voces humanas que fue lo que se midió, este micrófono captó una señal con más energía en variaciones esto puede deberse a que el micrófono esté más cerca de la persona o captó más ruido ambiental, en el micrófono 3 se observa un pico muy marcado en la zona de 100 Hz , lo que indica que esta frecuencia es la más fuerte en la señal captada esto sugiere que el micrófono registró un sonido con un tono grave predominante, que sería el golpe del que se habló anteriormente además del pico dominante, hay varias frecuencias con amplitudes más bajas que se extienden hasta los 3 kHz esto indica que el sonido no es completamente puro este micrófono muestra menos energía en frecuencias más altas esto puede indicar que captó una señal más enfocada en los tonos graves o que su respuesta en frecuencias altas es menor captó una señal con un tono grave predominante , con un fuerte pico en 100 Hz y algunos armónicos en el rango de 100 Hz-3 kHz .  

Con respecto a las gráficas de Densidad Espectral de Potencia todas siguen una tendencia descendente a medida que aumenta la frecuencia, la forma de la curva sugiere que hay un filtro natural o una limitación en el sistema que atenúa las frecuencias más altas analizando región de medias frecuencias (10² - 10³ Hz) evidenciamos que aquí es donde se observan diferencias más notables ya que en el MICRÓFONO 1 y MICRÓFONO 3, la curva muestra una ligera caída progresiva con pequeñas fluctuaciones mientras que El MICRÓFONO 2 mantiene una caída más uniforme y menos oscilaciones. El análisis entre región de altas frecuencias (10³ - 10⁴ Hz) se observa una caída más pronunciada en todos los micrófonos, lo que sugiere una pérdida de potencia en altas frecuencias debido a la respuesta del sistema o las características del micrófono y El MICRÓFONO 3 muestra más fluctuaciones en esta región antes de la caída final, cerca de 10⁴ Hz en todos los casos, hay una caída abrupta que indica el límite del ancho de banda del sistema. La representación logarítmica resalta que la energía está más concentrada en bajas frecuencias y que la atenuación en altas frecuencias es un patrón común en los tres micrófonos. Sin embargo, MICRÓFONO 2 parece ser el más estable en todo el espectro.  

 

![WhatsApp Image 2025-03-07 at 3 41 26 PM](https://github.com/user-attachments/assets/240332b1-839c-484d-af93-4dfa20bcf374)    
  |*Figura 10: Resultado de Componentes Independientes (ICA) y el Beamforming.*|     



## Licencia 
Open Data Commons Attribution License v1.0

## Temas:
# 📡 Procesamiento de Señales  
- Componentes Independientes (ICA)
- El Beamforming

# 🔊 Análisis en Frecuencia  
- Transformada de Fourier Discreta (DFT)
- Transformada Rápida de Fourier (FFT)

# 🖥️ Código e Implementación  
- Explicación del código  
- Ejecución y ejemplos  
- Mejoras y optimización  







