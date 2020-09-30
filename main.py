'''
LABORATORIO 3: MODULACION DE SEÃ‘ALES

Estudiante: Francisco Rousseau
Ayudante: Nicole Reyes
Profesor: Carlos GonzÃ¡lez

'''


######################## LIBRERIAS ########################
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

######################## FUNCIONES #######################

def Graficar(figura, titulo, xlab, ylab, x, y):
    plt.figure(figura, [7.5, 4.5])
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if 'Espectograma' in titulo:
        plt.specgram(x,Fs=y)
    else:
        plt.plot(x, y)
    plt.show()

#################### BLOQUE PRINCIPAL ####################

############ Parte 1 ############

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
Graficar("Grafico 1 - Amplitud", 'Audio original en el tiempo', "Tiempo [s]", "Amplitud [db]", Time, audio)

############ Parte 2 ############

# Se obtiene la Transformada de Fourier ¡Normalizada!
fourierY = fft(audio) / amplitudes

# Se obtiene la frecuencia de cada punto de la Transformada de Fourier
fourierX= fftfreq(int(amplitudes), 1.0/f)

''' Grafico - 2 '''
Graficar("Grafico 2 - Transformada", 'Transfromada de Fourier', "Frecuencia [Hz]", "Amplitud [dB]", fourierX, abs(fourierY))


''' Transformada de fourier Inversa '''

# Se obtiene la transfromada de fourier inversa y normaliza 
fourierInversaY = ifft(fourierY).real * amplitudes

''' Grafico - 3 '''
Graficar("Grafico 3 - Inversa", "Transfromada de Fourier Inversa", "Tiempo [s]", "Amplitud [dB]", Time, fourierInversaY)

############ Parte 3 ############

''' Grafico - 4 '''
Graficar("Grafico 4 - Espectograma", "Espectograma del audio", 'Tiempo [s]', 'Frecuencia [Hz]', audio, f)

############ Parte 4 ############

'''------> Filtro FIR <------'''
# Se define el rango en el que efectuara el filtro
ini = 300
end = 1200

# Se crea el filtro, de orden 6, aplicando las condiciones de un tipo de rango de pasa banda
# Se aplican 3 intervalos distitnos de frecuencia para el filtro tal como solicita el enunciado

b1, a1 = butter(N=6, Wn=[1 * ini/f, 1 * end/f], btype='band')
b2, a2 = butter(N=6, Wn=[2 * ini/f, 2 * end/f], btype='band')
b3, a3 = butter(N=6, Wn=[3 * ini/f, 3 * end/f], btype='band')

# Se filtran los audios
afiltrado1 = lfilter(b1, a1, audio)
afiltrado2 = lfilter(b2, a2, audio)
afiltrado3 = lfilter(b3, a3, audio)

# Se guardan los nuevos archivos de audio filtrados
wavfile.write("AudioFiltrado - 1.wav", f, afiltrado1.astype(np.int16)) #dtype=np.int16
wavfile.write("AudioFiltrado - 2.wav", f, afiltrado2.astype(np.int16)) #dtype=np.int16
wavfile.write("AudioFiltrado - 3.wav", f, afiltrado3.astype(np.int16)) #dtype=np.int16

''' Graficos - 5 '''
Graficar("Grafico 5 - Filtro 1", "Audio filtrado - 1", "Tiempo [s]", "Amplitud [dB]", Time, afiltrado1)
Graficar("Grafico 5 - Filtro 2", "Audio filtrado - 2", "Tiempo [s]", "Amplitud [dB]", Time, afiltrado2)
Graficar("Grafico 5 - Filtro 3", "Audio filtrado - 3", "Tiempo [s]", "Amplitud [dB]", Time, afiltrado3)

# Se obtienen las Tranformadas de Fourier ¡Normalizadas!
fourierY1 = fft(afiltrado1)/amplitudes
fourierY2 = fft(afiltrado2)/amplitudes
fourierY3 = fft(afiltrado3)/amplitudes

''' Graficos - 6 '''
Graficar("Grafico 6 - Transformada Filtrada 1", "Transfromada de Fourier Audio Filtrado - 1", "Frecuencia [Hz]", "Amplitud [dB]", fourierX, abs(fourierY1))
Graficar("Grafico 6 - Transformada Filtrada 2", "Transfromada de Fourier Audio Filtrado - 2", "Frecuencia [Hz]", "Amplitud [dB]", fourierX, abs(fourierY2))
Graficar("Grafico 6 - Transformada Filtrada 3", "Transfromada de Fourier Audio Filtrado - 3", "Frecuencia [Hz]", "Amplitud [dB]", fourierX, abs(fourierY3))

''' Transformada de fourier Inversa '''

#Se obtienen las transfromada de fourier inversas con la funcion ifft
fourierInversaY1 = ifft(fourierY1).real * amplitudes
fourierInversaY2 = ifft(fourierY2).real * amplitudes
fourierInversaY3 = ifft(fourierY3).real * amplitudes

''' Graficos - 7 '''
Graficar("Grafico 7 - Espectograma Filtrada 1", "Espectograma del Audio Filtrado - 1", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado1, f)
Graficar("Grafico 7 - Espectograma Filtrada 2", "Espectograma del Audio Filtrado - 2", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado2, f)
Graficar("Grafico 7 - Espectograma Filtrada 3", "Espectograma del Audio Filtrado - 3", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado3, f)

