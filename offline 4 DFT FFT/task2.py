import numpy as np
import matplotlib.pyplot as plt
import time
n=50

samples = np.arange(n) 
sampling_rate=100
# 1. Generate Random Discrete Signals
def generate_random_discrete_signal(n):
    return np.random.rand(n)

# 2. Implement DFT and IDFT
def DFT_computation(signal_a):
    length = len(signal_a)
    Dft = np.zeros(length, dtype=complex)
    for k in range(length):
        for n in range(length):
            Dft[k] += signal_a[n] * np.exp(-2j * np.pi * k * n / length)
    return Dft

def IDFT_computation(signal_a):
    length = len(signal_a)
    Idft = np.zeros(length, dtype=complex)
    for n in range(length):
        for k in range(length):
            Idft[n] += signal_a[k] * np.exp(2j * np.pi * k * n / length)
        Idft[n] /= length
    return Idft

def FFT(signal_a):
    length = len(signal_a)
    if length <= 1:
        return signal_a
    else:
        even = FFT(signal_a[0::2])
        odd = FFT(signal_a[1::2])
        odd = np.array(odd, dtype=complex)
        for k in range(length // 2):
            factor = np.exp(-2j * np.pi * k / length)
            odd[k] *= factor
        return np.concatenate([even + odd, even - odd])

def IFFT(signal_a):
    length = len(signal_a)
    if length <= 1:
        return signal_a
    else:
        even = IFFT(signal_a[0::2])
        odd = IFFT(signal_a[1::2])
        odd = np.array(odd, dtype=complex)
        for k in range(length // 2):
            factor = np.exp(2j * np.pi * k / length)
            odd[k] *= factor
        return np.concatenate([even + odd, even - odd]) / length

# 3. Measure runtime
def measure_runtime(signal, func, repetitions=10):
    total_time = 0
    for _ in range(repetitions):
        start_time = time.time()
        func(signal)
        total_time += time.time() - start_time
    return total_time / repetitions

def main():
    sample_sizes = [2**k for k in range(2, 10)]  # Powers of 2 from 4 to 512
    dft_times = []
    fft_times = []
    idft_times = []
    ifft_times = []
    
    for n in sample_sizes:
        signal = generate_random_discrete_signal(n)

        # Measure runtimes for DFT and FFT
        dft_time = measure_runtime(signal, DFT_computation)
        fft_time = measure_runtime(signal, FFT)
        
        # Measure runtimes for IDFT and IFFT
        dft_signal = DFT_computation(signal)  
        idft_time = measure_runtime(dft_signal, IDFT_computation)

        fft_signal = FFT(signal)  
        ifft_time = measure_runtime(fft_signal, IFFT)

        # Store results
        dft_times.append(dft_time)
        fft_times.append(fft_time)
        idft_times.append(idft_time)
        ifft_times.append(ifft_time)

    # Plot runtime comparison: DFT vs FFT
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, dft_times, label="DFT", marker="o")
    plt.plot(sample_sizes, fft_times, label="FFT", marker="o")
    plt.xscale("log",base=2)
    plt.yscale("log")
    plt.xlabel("Signal Size (n) (log scale)")
    plt.ylabel("Average Runtime (s) (log scale)")
    plt.title("Runtime Comparison: DFT vs FFT")
    plt.legend()
    plt.grid (True)
    # plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # Plot runtime comparison: IDFT vs IFFT
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, idft_times, label="IDFT", marker="o")
    plt.plot(sample_sizes, ifft_times, label="IFFT", marker="o")
    plt.xscale("log",base=2)
    plt.yscale("log")
    plt.xlabel("Signal Size (n) (log scale)")
    plt.ylabel("Average Runtime (s) (log scale)")
    plt.title("Runtime Comparison: IDFT vs IFFT")
    plt.legend()
    plt.grid (True)
    # plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(sample_sizes, idft_times/ifft_times, label="DFT", marker="o")
    # # plt.plot(sample_sizes, fft_times, label="FFT", marker="o")
    # plt.xscale("log",base=2)
    # plt.yscale("log")
    # plt.xlabel("Signal Size (n) (log scale)")
    # plt.ylabel("Average Runtime (s) (log scale)")
    # plt.title("Runtime Comparison: DFT vs FFT")
    # plt.legend()
    # plt.grid (True)
    # # plt.grid(which="both", linestyle="--", linewidth=0.5)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(sample_sizes, dft_times/fft_times, label="DFT", marker="o")
    # # plt.plot(sample_sizes, fft_times, label="FFT", marker="o")
    # plt.xscale("log",base=2)
    # plt.yscale("log")
    # plt.xlabel("Signal Size (n) (log scale)")
    # plt.ylabel("Average Runtime (s) (log scale)")
    # plt.title("Runtime Comparison: DFT vs FFT")
    # plt.legend()
    # plt.grid (True)
    # # plt.grid(which="both", linestyle="--", linewidth=0.5)
    # plt.show()

    print("Task completed successfully!")


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

# main()


signal_a= np.random.rand(32)
print((signal_a))
plot_signal(signal_a,"signal a ")
plot_signal(IFFT(FFT(signal_a)),"reconstructed signal a ")



# plt.figure(figsize=(10, 6))
# plt.stem(,signal_a)
