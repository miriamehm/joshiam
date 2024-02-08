#Forberedelse oppgave 5.1

#Beskriv med Python-kode hvordan krysskorrelasjon kan brukes for å finne effektiv forsinkelse mellom
#to lydsignaler som har samplingsfrekvensen fs.
#Hint: du skal finne for hvilken forsinkelse som numpy.abs(krysskorrelasjonsfunksjonen) har
#maksimum.

import numpy as np
import scipy.signal as si
import matplotlib.pyplot as plt


#Defining signals
sampling_freq = 100
signal_x = np.array([1,5,3,7,8,2,4,7,0,0])  #length = 10
signal_y = np.array([0,0,1,5,3,7,8,2,4,7])

#Finding autocorrelation
cross_xy = si.correlate(signal_x,signal_y)

#Finding l for largest value in the autocorrelation
abs_cross_xy = np.abs(cross_xy)
l_index = np.argmax(abs_cross_xy)

#Calculating delay between signals
t_delay = l_index / sampling_freq

#print(l_index)
#print(t_delay)

# Generating time axis
time = np.arange(-len(signal_x) + 1, len(signal_x))

print(time[l_index])

# Plotting autocorrelation
plt.figure()
plt.stem(time, cross_xy)
plt.xlim(time[0] - 2, time[-1] + 2)  # Extend x-axis by 1 unit on each side
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Autocorrelation')
plt.show()

