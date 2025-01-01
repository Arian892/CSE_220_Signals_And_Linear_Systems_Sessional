import numpy as np
import matplotlib.pyplot as plt
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000



#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    # shift_samples = -9
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B

#implement other functions and logic
def DFT_computation(signal_a):
    length = len(signal_a)
    Dft = np.zeros (length , dtype = complex)   
    for k in range  (length):
        for n in range (length):
            Dft[k] += signal_a[n] * np.exp(-2j * np.pi * k * n / length)
    return Dft 

    
def IDFT_computation (signal_a):
    length = len(signal_a)
    Idft = np.zeros (length , dtype = complex)
    for n in range (length):
        for k in range (length):
            Idft[n] += signal_a[k] * np.exp(2j * np.pi * k * n / length)
        Idft[n] = Idft[n] / length
    return Idft    


def cross_correlation_calculation(signal_a, signal_b):
    dft_a = DFT_computation(signal_a)
    dft_b = DFT_computation(signal_b)
    conjugate_dft_b = np.conj(dft_b)
    cross_draft = np.zeros(len(dft_a), dtype=complex)
    cross_draft = dft_a * conjugate_dft_b
    cross_correlation = IDFT_computation(cross_draft)
    
    return np.roll(cross_correlation.real, len(cross_correlation) // 2)



def sample_lag_calculation(signal_a, signal_b):
    cross_correlation = cross_correlation_calculation(signal_a, signal_b)
    max_index = np.argmax(cross_correlation) - len(cross_correlation) // 2
    return -max_index


def Distance_estimation(signal_a, signal_b):
    sample_lag = sample_lag_calculation(signal_a, signal_b)
    distance = abs(sample_lag) / sampling_rate * wave_velocity
    return distance



def plot_signal(signal, title="Signal Plot", xlabel="Sample Index", ylabel="Amplitude", color="blue"):

    """
    Plots the time-domain signal.


    """
    n = len(signal)
    x = np.arange(n)
    
    # Plot the signal
    plt.figure(figsize=(10, 6))
    plt.stem(x, signal, basefmt=" ", linefmt=color, markerfmt=color)  # Corrected format for markerfmt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_signal_frequency(signal, title="Signal Plot", xlabel="Frequency Index", ylabel="magnitude", color="blue"):
    """
    Plots the frequency-domain signal.


    """
    n = len(signal)
    x = np.arange(n)
    
    # Plot the signal
    plt.figure(figsize=(10, 6))
    plt.stem(x, np.abs(signal), basefmt=" ", linefmt=color, markerfmt=color)  # Corrected format for markerfmt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()



def plot_cross_correlation (signal,title="Signal Plot", xlabel="Sample Index", ylabel="Amplitude", color="blue"):

    """
    Plots the time-domain signal.


    """
    n = len(signal)
    x = np.arange(-n//2 , n//2)
    
    # Plot the signal
    plt.figure(figsize=(10, 6))
    plt.stem(x, signal, basefmt=" ", linefmt=color, markerfmt=color)  # Corrected format for markerfmt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()



signal_a, signal_b = generate_signals(5)


plot_signal(signal_a, title="Signal A", color="blue")
plot_signal_frequency (DFT_computation(signal_a), title="Frequency Spectrum Signal A", color="blue")
plot_signal(signal_b, title="Signal B", color="red")
plot_signal_frequency (DFT_computation(signal_b), title="Frequency Spectrum Signal B", color="red")

cross_correlation = cross_correlation_calculation(signal_b, signal_a)
plot_cross_correlation(cross_correlation, title="Correlation",xlabel="Lag (samples)", color="green")
print ("lag ", sample_lag_calculation(signal_b, signal_a))
print ("Distance ", Distance_estimation(signal_b, signal_a))