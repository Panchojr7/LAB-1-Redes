#Autor: Francisco Rousseau Riveros


''' Importaciones '''
import scipy
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

################### Parte 1 ###################

''' Extrayendo senal de audio '''
# f -> frecuencia de handel.wav
# audio -> arreglo de amplitudes de handel.wav y tipo de dato
f,audio = wavfile.read("handel.wav")
print(f)

# Numero de amplitudes dentro de Handel.wav
amplitudes = len(audio)

# Arreglo con valores de tiempo para cada amplitud
Time = np.linspace(0, amplitudes/f, amplitudes) 

''' Grafico - 1 '''
plt.figure("Grafico 1 - Amplitud", [7.5, 4.5])
plt.title('Audio original en el tiempo')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(Time, audio)

################### Parte 2 ###################

# Se obtiene la Transformada de Fourier ¡Normalizada!
fourierY = fft(audio) / amplitudes

# Se obtiene la frecuencia de cada punto de la Transformada de Fourier
fourierX= fftfreq(int(amplitudes), 1.0/f)

''' Grafico - 2 '''
plt.figure("Grafico 2 - Transformada", [7.5, 4.5])
plt.title("Transfromada de Fourier")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.plot(fourierX, abs(fourierY))
#plt.plot(fourierX, fourierY) ## <- para la version completa (grafico espejo en eje y) descomentar

''' Transformada de fourier Inversa '''

# Se obtiene la transfromada de fourier inversa y normaliza 
fourierInversaY = ifft(fourierY).real * amplitudes

''' Grafico - 3 '''
plt.figure("Grafico 3 - Inversa", [7.5, 4.5])
plt.title("Transfromada de Fourier Inversa")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(Time, fourierInversaY)

################### Parte 3 ###################

''' Grafico - 4 '''
plt.figure("Grafico 4 - Espectograma", [7.5, 4.5])
plt.title("Espectograma del audio")
plt.specgram(audio,Fs=f)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')

################### Parte 4 ###################

'''------> Filtro FIR <------'''
# Se define el rango en el que efectuara el filtro
ini = 300
end = 1200

# Se crea el filtro, de orden 6, aplicando las condiciones de un tipo de rango de pasa banda
b, a = butter(N=6, Wn=[2 * ini/f, 2 * end/f], btype='band')

# Se filtra el audio
afiltrado = lfilter(b, a, audio)

# Se guarda el nuevo archivo de audio filtrado
wavfile.write("AudioFiltrado.wav", f, afiltrado.astype(np.int16)) #dtype=np.int16

''' Grafico - 5 '''
plt.figure("Grafico 5 - Filtro", [7.5, 4.5])
plt.title("Audio filtrado")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(Time, afiltrado)

# Obtencion de la Tranformada de Fourier ¡Normalizada!
fourierY = fft(afiltrado)/amplitudes

''' Grafico - 6 '''
plt.figure("Grafico 6 - Transformada Filtrada", [7.5, 4.5])
plt.title("Transfromada de Fourier Audio Filtrado")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.plot(fourierX,abs(fourierY))

''' Transformada de fourier Inversa '''

#Se transforma la transfromada de fourier con la funcion ifft
fourierInversaY= ifft(fourierY).real * amplitudes

''' Grafico - 7 '''
plt.figure("Grafico 7 - Espectograma Filtrada", [7.5, 4.5])
plt.title("Espectograma del Audio Filtrado")
plt.specgram(afiltrado,Fs=f)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')

plt.show()

