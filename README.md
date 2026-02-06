# Ideal, Natural, & Flat-top -Sampling
# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Tools required
Collab
# Program
# Ideal Sampling
```
#Impulse Sampling
 import numpy as np
 import matplotlib.pyplot as plt
 from scipy.signal import resample
 fs = 100
 t = np.arange(0, 1, 1/fs) 
 f = 5
 signal = np.sin(2 * np.pi * f * t)
 plt.figure(figsize=(10, 4))
 plt.plot(t, signal, label='Continuous Signal')
 plt.title('Continuous Signal (fs = 100 Hz)')
 plt.xlabel('Time [s]')
 plt.ylabel('Amplitude')
 plt.grid(True)
 plt.legend()
 plt.show()
 t_sampled = np.arange(0, 1, 1/fs)
 signal_sampled = np.sin(2 * np.pi * f * t_sampled)
 plt.figure(figsize=(10, 4))
 plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
 plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
 plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
 plt.xlabel('Time [s]')
 plt.ylabel('Amplitude')
 plt.grid(True)
plt.legend()
 plt.show()
 reconstructed_signal = resample(signal_sampled, len(t))
 plt.figure(figsize=(10, 4))
 plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
 plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
 plt.ylabel('Amplitude')
 plt.grid(True)
 plt.legend()
 plt.show()tach the program
```

# Natural Sampling
```
# Natural Sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000      # Sampling frequency (Hz)
T = 1          # Duration (seconds)
t = np.arange(0, T, 1/fs)  # Time vector

# Message Signal
fm = 5  # Message frequency (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)

# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1

# Natural Sampling
nat_signal = message_signal * pulse_train

# Sampled signal during pulses
sampled_signal = nat_signal[pulse_train == 1]

# Time instants of samples
sample_times = t[pulse_train == 1]

# Zero Order Hold Reconstruction
reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index + pulse_width] = sampled_signal[i]

# Low-pass filter
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

# Plotting
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

# Flat Top Sampling
```
# Flat-top Sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000      # Sampling frequency (Hz)
T = 1          # Duration (seconds)
t = np.arange(0, T, 1/fs)

# Message signal
fm = 5         # Message frequency (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Sampling parameters
pulse_rate = 50  # pulses per second
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))

# Ideal impulse sampling (for reference)
pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1

# Flat-top sampled signal
flat_top_signal = np.zeros_like(t)
pulse_width_samples = int(fs / (2 * pulse_rate))  # flat-top width

# Flat-top sampling process
for index in pulse_train_indices:
    sample_value = message_signal[index]
    end_index = min(index + pulse_width_samples, len(t))
    flat_top_signal[index:end_index] = sample_value

# Low-pass filter for reconstruction
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

# Plotting
plt.figure(figsize=(14, 10))

# Original message signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Ideal sampling instants
plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices],
         pulse_train[pulse_train_indices],
         basefmt=" ",
         label='Ideal Sampling Instants')
plt.legend()
plt.grid(True)

# Flat-top sampled signal
plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.legend()
plt.grid(True)

# Reconstructed signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal,
         label='Reconstructed Signal',
         color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

# Output Waveform
# Ideal Sampling
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/04c8a8d3-6aa1-468f-b9d3-c671543bf968" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/7dc37cf5-b713-4e74-9534-c02858c350b7" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/b6c4cb60-b236-4043-a30a-a4b782c8420e" />

# Natural Sampling
<img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/45215384-d8e9-4884-a6e3-5b316518c19a" />

# Flat top Sampling
<img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/811d98fa-4f47-4fdd-89d1-1822fcdc624a" />


# Results
Thus, the construction and reconstruction of Ideal, Natural, and Flat-top sampling were successfully implemented using Python, and the corresponding waveforms were obtained.


