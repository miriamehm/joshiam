import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_channels(data):
    plt.figure(figsize=(12, 8))  # Create figure outside the loop
    for i in range(data.shape[1]):  # Use data.shape[1] instead of data_multi.shape[1]
        plt.subplot(data.shape[1], 1, i+1)
        plt.plot(np.arange(data.shape[0]) * sample_period, data[:, i])  # Use data instead of data_multi
        plt.xlim([0.2, 0.22])
        plt.ylim([-0.05, 1.3])
        plt.title(f'Channel {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()  # Call plt.show() after the loop has completed


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (samples, channels) float64 array of
    sampled data from all channels channels.

    Example (requires a recording named foo.bin):
    
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" .astype('float64') casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data


# Import data from bin file

sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1
        else 'number1.bin')

print(data)
data_multi = data * 0.81*10**(-3)
print(data_multi)

def apply_hanning_window(signal):
    window = np.hanning(len(signal))
    return signal * window

def apply_rect_window(signal):
    window = np.ones(len(signal))
    return signal * window

def apply_hamming_window(signal):
    window = np.hamming(len(signal))
    return signal * window

def apply_gaussian_window(signal):
    window = np.gaussian(len(signal))
    return signal * window

def plot_fft(data, sample_period, zero_padding_factor=2):
    num_channels = data.shape[1]
    time_axis = np.arange(0, data.shape[0] * sample_period, sample_period)

    plt.figure(figsize=(12, 8))
    for i in range(num_channels):
        channel_data = data[:, 2]

        # Apply Hanning window and perform FFT with zero-padding
        channel_data = data[:, i]
        fft_result = np.fft.fft(channel_data, n=len(channel_data) * zero_padding_factor)
        fft_freq = np.fft.fftfreq(len(fft_result), sample_period)
        
        # FFT without Hanning window
        fft_result_no_window = np.fft.fft(channel_data, n=len(channel_data) * zero_padding_factor)
        fft_magnitude_no_window = np.abs(fft_result_no_window) / len(channel_data)
        fft_magnitude_dB_no_window = 20 * np.log10(fft_magnitude_no_window)
        
        # FFT with Hanning window
        channel_data_windowed = channel_data * apply_hanning_window(channel_data)
        fft_result_windowed = np.fft.fft(channel_data_windowed, n=len(channel_data_windowed) * zero_padding_factor)
        fft_magnitude_windowed = np.abs(fft_result_windowed) / len(channel_data_windowed)
        fft_magnitude_dB_windowed = 20 * np.log10(fft_magnitude_windowed)
        
        plt.title(f'Channel {i+1} FFT with and without Hanning Window')
        
        if i == 3:
            plt.xlim(900, 1100)
        elif i == 4:
            plt.xlim(4900, 5100)

        plt.plot(fft_freq, fft_magnitude_dB_no_window, label='Without Hanning Window', color='blue')
        plt.plot(fft_freq, fft_magnitude_dB_windowed, label='With Hanning Window', color='red')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.ylim(-150, 5)

    plt.tight_layout()
    plt.show()

def plot_fft_3(data, sample_period, zero_padding_factor=2):
    # Select data for channel 3
    channel_data = data[:, 2]  # Channel 3 is at index 2

    # Apply FFT
    fft_result = np.fft.fft(channel_data)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude_dB = 20 * np.log10(fft_magnitude)
    max_amplitude = np.max(fft_magnitude_dB)

    # Frequency axis
    fft_freq = np.fft.fftfreq(len(fft_result), sample_period)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(fft_freq, fft_magnitude_dB-max_amplitude, color='blue')
    plt.title('FFT of 2000 Hz Sine Wave')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.xlim(1900, 2100)  # Limit x-axis to the desired frequency range
    plt.ylim(-100, 0)  # Limit y-axis for better visualization
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_window_fft(data, sample_period, zero_padding_factor=2):
    # Select data for channel 3
    channel_data = data[:, 2]  # Channel 3 is at index 2

    # Apply Hanning window and perform FFT with zero-padding
    fft_result_no_window = np.fft.fft(channel_data, n=len(channel_data) * zero_padding_factor)
    fft_magnitude_no_window = np.abs(fft_result_no_window) / len(channel_data)
    fft_magnitude_dB_no_window = 20 * np.log10(fft_magnitude_no_window)

    channel_data_windowed = channel_data * apply_rect_window(channel_data)
    fft_result_windowed = np.fft.fft(channel_data_windowed, n=len(channel_data_windowed) * zero_padding_factor)
    fft_magnitude_windowed = np.abs(fft_result_windowed) / len(channel_data_windowed)
    fft_magnitude_dB_windowed = 20 * np.log10(fft_magnitude_windowed)

    # Frequency axis
    fft_freq = np.fft.fftfreq(len(fft_result_no_window), sample_period)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.title('2000 Hz Sine Wave FFT with and without Rectangular Window')

    plt.plot(fft_freq, fft_magnitude_dB_no_window, label='Without Rectangular Window', color='blue')
    plt.plot(fft_freq, fft_magnitude_dB_windowed, label='With Rectangular Window', color='red')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.xlim(1900, 2100)  # Limit x-axis to the desired frequency range
    plt.ylim(-150, 5)  # Limit y-axis for better visualization

    plt.tight_layout()
    plt.show()

def plot_fft_loop(data, sample_period, zero_padding_factor=2):
    num_channels = data.shape[1]

    plt.figure(figsize=(12, 8))
    for i in range(num_channels):
        # Select data for the current channel
        channel_data = data[:, i]

        # Apply FFT
        fft_result = np.fft.fft(channel_data)
        fft_magnitude = np.abs(fft_result)
        fft_magnitude_dB = 20 * np.log10(fft_magnitude)
        max_amplitude = np.max(fft_magnitude_dB)

        # Frequency axis
        fft_freq = np.fft.fftfreq(len(fft_result), sample_period)

        # Create subplot for each channel
        plt.subplot(num_channels, 1, i+1)
        plt.plot(fft_freq, fft_magnitude_dB-max_amplitude, color='blue')
        plt.title(f'FFT of Channel {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.xlim(1900, 2100)  # Limit x-axis to the desired frequency range
        plt.ylim(-100, 0)  # Limit y-axis for better visualization
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_window_fft_loop(data, sample_period, zero_padding_factor=2):
    num_channels = data.shape[1]

    plt.figure(figsize=(12, 8))
    for i in range(num_channels):
        # Select data for the current channel
        channel_data = data[:, i]

        # Apply FFT without rectangular window
        fft_result_no_window = np.fft.fft(channel_data, n=len(channel_data) * zero_padding_factor)
        fft_magnitude_no_window = np.abs(fft_result_no_window) / len(channel_data)
        fft_magnitude_dB_no_window = 20 * np.log10(fft_magnitude_no_window)

        # Apply rectangular window and perform FFT
        channel_data_windowed = channel_data * apply_hamming_window(channel_data)
        fft_result_windowed = np.fft.fft(channel_data_windowed, n=len(channel_data_windowed) * zero_padding_factor)
        fft_magnitude_windowed = np.abs(fft_result_windowed) / len(channel_data_windowed)
        fft_magnitude_dB_windowed = 20 * np.log10(fft_magnitude_windowed)

        # Frequency axis
        fft_freq = np.fft.fftfreq(len(fft_result_no_window), sample_period)

        # Create subplot for each channel
        plt.subplot(num_channels, 1, i+1)
        plt.plot(fft_freq, fft_magnitude_dB_no_window, label='Without Hamming Window', color='blue')
        plt.plot(fft_freq, fft_magnitude_dB_windowed, label='With Hamming Window', color='red')
        plt.title(f'Channel {i+1} FFT with and without Hamming Window')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.xlim(1975, 2025)  # Limit x-axis to the desired frequency range
        plt.ylim(-80, 5)  # Limit y-axis for better visualization

    plt.legend()
    plt.tight_layout()
    plt.show()



# Plot FFTs with Hanning window and zero-padding for channels 4 and 5
plot_window_fft_loop(data_multi, sample_period)
#plot_window_fft(data_multi, sample_period)
#plot_channels(data_multi)