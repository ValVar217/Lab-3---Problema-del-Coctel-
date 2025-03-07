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
▫️**os**: Nos permite la gestion de rutas y archivos.  
▫️**librosa**: Es una biblioteca utilizada para la carga, análisis y manipulación de señales de audio.    
▫️**librosa.display**: Se utiliza para visualizar las señales de audio en forma de espectrogramas y formas de onda.   
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


![WhatsApp Image 2025-03-07 at 3 41 08 PM](https://github.com/user-attachments/assets/32373e9b-767f-445f-bac5-735f3523ecb6)    
  |*Figura 4: Señal del Microfono 1.*|    

![WhatsApp Image 2025-03-07 at 3 41 08 PM (1)](https://github.com/user-attachments/assets/d295e268-a5a3-4bce-97dc-edd6a5423184)    
  |*Figura 5: Señal del Microfono 2.*|     

![WhatsApp Image 2025-03-07 at 3 41 08 PM (2)](https://github.com/user-attachments/assets/63da823f-e528-4469-a2da-9ac1108d23a7)  
  |*Figura 6: Señal del Microfono 3.*|   

  





