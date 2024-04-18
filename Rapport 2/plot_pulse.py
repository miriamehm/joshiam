import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal



data = np.loadtxt('/Users/joshjude/Documents/skole/semester 6/ttt4280 sensor/lab1-kopi/joshiam/Lab3/RGBmeasurementswarmer.txt')
data = signal.detrend(data)



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

def plotFFT():

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

def plotPeriodogram_with_filter():
    plt.figure(figsize=(10, 6))
    lowcut = 0.5  
    highcut = 3.3  
    fs = 1 / (time[1] - time[0])  

    red_filtered = butter_bandpass_filter(red, lowcut, highcut, fs)
    green_filtered = butter_bandpass_filter(green, lowcut, highcut, fs)
    blue_filtered = butter_bandpass_filter(blue, lowcut, highcut, fs)

    f_red, Pxx_den_red = signal.periodogram(red_filtered, fs=fs)
    f_green, Pxx_den_green = signal.periodogram(green_filtered, fs=fs)
    f_blue, Pxx_den_blue = signal.periodogram(blue_filtered, fs=fs)

    Pxx_den_red_dB = 10 * np.log10(Pxx_den_red)
    Pxx_den_green_dB = 10 * np.log10(Pxx_den_green)
    Pxx_den_blue_dB = 10 * np.log10(Pxx_den_blue)

    plt.subplot(3, 1, 1)
    peak_freq_red = f_red[np.argmax(Pxx_den_red)]
    normalizedRed = Pxx_den_red_dB - np.max(Pxx_den_red_dB)
    plt.plot(f_red, normalizedRed, color='red', label='Rød kanal')
    plt.xlim(0.25, 5)
    plt.ylim(-50, 10)
    plt.grid(True)
    plt.plot(peak_freq_red, np.max(normalizedRed), marker='o', markersize=5, color='red', label=f'Frekvenstopp: {peak_freq_red:.3f} Hz')
    plt.title('Periodogram - rød kanal')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Relativ effekt [dB]')
    plt.legend()

    plt.subplot(3, 1, 2)
    peak_freq_green = f_green[np.argmax(Pxx_den_green)]
    normalizedGreen = Pxx_den_green_dB - np.max(Pxx_den_green_dB)
    plt.plot(f_green, normalizedGreen, color='green', label='Grønn kanal')
    plt.xlim(0.25, 5)
    plt.ylim(-50, 10)
    plt.grid(True)
    plt.plot(peak_freq_green, np.max(normalizedGreen), marker='o', markersize=5, color='green', label=f'Frekvenstopp: {peak_freq_green:.3f} Hz')
    plt.title('Periodogram - grønn kanal')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Relativ effekt [dB]')
    plt.legend()

    plt.subplot(3, 1, 3)
    peak_freq_blue = f_blue[np.argmax(Pxx_den_blue)]
    normalizedBlue = Pxx_den_blue_dB - np.max(Pxx_den_blue_dB)
    plt.plot(f_blue, normalizedBlue, color='blue', label='Blå kanal')
    plt.xlim(0.25, 5)
    plt.ylim(-50, 10)
    plt.grid(True)
    plt.plot(peak_freq_blue, np.max(normalizedBlue), marker='o', markersize=5, color='blue', label=f'Peak: {peak_freq_blue:.3f} Hz')
    plt.title('Periodogram - blå kanal')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Relativ effekt [dB]')
    plt.legend()

    plt.tight_layout()
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

plotPeriodogram_with_filter()


