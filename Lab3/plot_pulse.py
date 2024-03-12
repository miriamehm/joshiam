import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci

data = np.loadtxt('joshiam\Lab3\RGBmeasurements2.txt')
data = sci.detrend(data)



red = data[:, 0][3:]
green = data[:, 1][3:]
blue = data[:, 2][3:]

time = np.linspace(0,30,len(red))

redFFT = np.fft.fft(red)
greenFFT = np.fft.fft(green)
blueFFT = np.fft.fft(blue)

frequencyBin = np.fft.fftfreq(len(time), time[1]-time[0])

def plotData() :
    plt.figure(figsize=(10,6))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, 20*np.log10(red), color='red')
    plt.title('Rød')
    plt.xlabel('Tid[s]')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(time, 20*np.log10(np.abs(green)), color='green')
    plt.title('Grønn')
    plt.xlabel('Tid[s]')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(time, 20*np.log10(blue), color='blue')
    plt.title('Blå')
    plt.xlabel('Blå[s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def plotFFT() :

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(frequencyBin, np.abs(redFFT), color='red')
    plt.title('FFT av rød fargekanal')
    plt.xlabel('Frekvens [Hz]')
    plt.xlim(-2,2)
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(frequencyBin, np.abs(greenFFT), color='green')
    plt.title('FFT av grønn fargekanal')
    plt.xlabel('Frekvens [Hz]')
    plt.xlim(-2,2)
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(frequencyBin, np.abs(blueFFT), color='blue')
    plt.title('FFT av blå fargekanal')
    plt.xlabel('Frekvens [Hz]')
    plt.xlim(-2,2)
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


plotFFT()
