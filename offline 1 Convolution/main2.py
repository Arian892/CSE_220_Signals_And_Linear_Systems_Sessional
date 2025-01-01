import numpy as np
import matplotlib.pyplot as plt
from ContinousSignal import ContinousSignal
from LTI_Continous import LTI_Continous

def plot_continous_signals(input_signal, impulse_response):
    img_root = 'continous_signal_images'
    delta = 0.5  # You can change this value as needed
    
    # Create an LTI system with the given impulse response
    lti = LTI_Continous(impulse_response)

    # Get unit impulses from linear combination
    unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)
    
    # Calculate cumulative sum of unit impulses directly
    cumulative_sum_impulses = ContinousSignal(lambda t: 0, input_signal.INF)
    for impulse in unit_impulses:
        cumulative_sum_impulses = cumulative_sum_impulses.add(impulse)
    
    # Prepare for plotting
    total_plots = len(unit_impulses) + 1  # Adding one for cumulative sum
    fig1, axs1 = plt.subplots((total_plots + 2) // 3, 3, figsize=(15, 3 * ((total_plots + 2) // 3)))
    axs1 = axs1.flatten()

    # Adjust time array range based on INF and delta
    time_array = np.linspace(-input_signal.INF / delta, input_signal.INF / delta, 1000)
    
    # Plot unit impulses
    for idx, unit_impulse in enumerate(unit_impulses):
        axs1[idx].plot(time_array, unit_impulse.evaluate(time_array))
        time_index = idx - int(unit_impulse.INF/delta)
        axs1[idx].set_title(f'&(t-({time_index}∇))x({time_index}∇)∇')  # Updated title format
        axs1[idx].set_xlabel('t (Time Index)')
        axs1[idx].set_ylabel('x(t)')
        axs1[idx].grid(True)

    # Plot cumulative sum of unit impulses
    axs1[len(unit_impulses)].plot(time_array, cumulative_sum_impulses.evaluate(time_array))
    axs1[len(unit_impulses)].set_title('Cumulative Sum (Impulses)')  # You can adjust this title as needed
    axs1[len(unit_impulses)].set_xlabel('t (Time Index)')
    axs1[len(unit_impulses)].set_ylabel('x(t)')
    axs1[len(unit_impulses)].grid(True)

    # Hide any extra subplots
    for idx in range(len(unit_impulses) + 1, len(axs1)):
        axs1[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{img_root}/linear_combination_impulses.png')
    # plt.show()

    # Calculate and plot output signals
    output_unit_impulses = lti.output_approx(input_signal, delta)
    cumulative_output_sum = ContinousSignal(lambda t: 0, input_signal.INF)
    for output_component in output_unit_impulses:
        cumulative_output_sum = cumulative_output_sum.add(output_component)
    
    # Prepare for output plotting
    total_output_plots = len(output_unit_impulses) + 1  # Adding one for cumulative output
    fig2, axs2 = plt.subplots((total_output_plots + 2) // 3, 3, figsize=(15, 3 * ((total_output_plots + 2) // 3)))
    axs2 = axs2.flatten()
    
    time_array_output = np.linspace(-input_signal.INF , input_signal.INF , 1000)
    
    # Plot output components
    for idx, component in enumerate(output_unit_impulses):
        axs2[idx].plot(time_array_output, component.evaluate(time_array_output))
        time_index = idx - int(component.INF/delta)
        axs2[idx].set_title(f'h(t-({time_index}∇))x({time_index}∇)∇')  # Updated title format
        axs2[idx].set_xlabel('t (Time Index)')
        axs2[idx].set_ylabel('x(t)')
        axs2[idx].grid(True)

    # Plot cumulative sum of output components
    axs2[len(output_unit_impulses)].plot(time_array_output, cumulative_output_sum.evaluate(time_array_output))
    axs2[len(output_unit_impulses)].set_title('Cumulative Sum (Output)')  # You can adjust this title as needed
    axs2[len(output_unit_impulses)].set_xlabel('t (Time Index)')
    axs2[len(output_unit_impulses)].set_ylabel('x(t)')
    axs2[len(output_unit_impulses)].grid(True)

    # Hide any extra subplots
    for idx in range(len(output_unit_impulses) + 1, len(axs2)):
        axs2[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{img_root}/output_components.png')
    # plt.show()
def plot_linear_sums(input_signal, impulse_response):
    img_root = 'continous_signal_images'
    deltas = [0.5, 0.1, 0.05, 0.01]  # Different delta values

    # Prepare for plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    # Prepare time array for the first delta value for consistent x-axis limits
    delta_first = deltas[0]
    time_array_first = np.linspace(-input_signal.INF / delta_first, input_signal.INF / delta_first, 1000)

    # Loop through different delta values
    for idx, delta in enumerate(deltas):
        # Create an LTI system with the given impulse response
        lti = LTI_Continous(impulse_response)

        # Get unit impulses from linear combination
        unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)

        # Calculate cumulative sum of unit impulses directly
        cumulative_sum_impulses = ContinousSignal(lambda t: 0, input_signal.INF)
        for impulse in unit_impulses:
            cumulative_sum_impulses = cumulative_sum_impulses.add(impulse)

        # Prepare time array for plotting
        time_array = np.linspace(-input_signal.INF , input_signal.INF , 1000)

        # Plot input signal and cumulative sum for this delta in the respective subplot
        axs[idx].plot(time_array_first, input_signal.evaluate(time_array_first), label='Input Signal', color='black', linewidth=2)
        axs[idx].plot(time_array_first, cumulative_sum_impulses.evaluate(time_array_first), label=f'Cumulative Sum (Δ={delta})')

        # Set titles and labels
        axs[idx].set_title(f'Delta = {delta}')
        axs[idx].set_xlabel('t (Time Index)')
        axs[idx].set_ylabel('x(t)')
        axs[idx].legend()
        axs[idx].grid(True)

    # Set the same x-axis limits for all subplots based on the first delta
    axs[0].set_xlim(time_array_first[0], time_array_first[-1])
    for ax in axs:
        ax.set_xlim(axs[0].get_xlim())

    plt.suptitle('Cumulative Sum of Linear Signals with Different Δ Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    plt.savefig(f'{img_root}/linear_sum_comparison.png')
    # plt.show()
def plot_output_sum(input_signal, impulse_response):
    img_root = 'continous_signal_images'
    deltas = [0.5, 0.1, 0.05, 0.01]

    # Prepare for plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    delta_first = deltas[0]
    time_array_first = np.linspace(-input_signal.INF / delta_first, input_signal.INF / delta_first, 1000)

    # Loop through different delta values
    for idx, delta in enumerate(deltas):
        lti = LTI_Continous(impulse_response)
        output_unit_impulses = lti.output_approx(input_signal, delta)

        # Calculate cumulative sum of output components
        cumulative_output_sum = ContinousSignal(lambda t: 0, input_signal.INF)
        for output_component in output_unit_impulses:
            cumulative_output_sum = cumulative_output_sum.add(output_component)

        # Prepare time array for plotting
        time_array = np.linspace(-input_signal.INF / delta, input_signal.INF / delta, 1000)

        # Plot cumulative output sum for this delta in the respective subplot
        axs[idx].plot(time_array, cumulative_output_sum.evaluate(time_array), label='Cumulative Output', color='blue')
        
        # Plot the new function y(t) = (1 - e^(-t)) u(t)
        new_func = ContinousSignal(lambda t: (1 - np.exp(-t)) * (t >= 0), input_signal.INF)
        axs[idx].plot(time_array, new_func.evaluate(time_array), label='y(t) = (1 - e^(-t))u(t)', color='orange')

        axs[idx].set_title(f'Cumulative Output Sum (Δ={delta})')
        axs[idx].set_xlabel('t (Time Index)')
        axs[idx].set_ylabel('x(t)')
        axs[idx].grid(True)
        axs[idx].legend()

    axs[0].set_xlim(time_array_first[0], time_array_first[-1])
    for ax in axs:
        ax.set_xlim(axs[0].get_xlim())

    plt.suptitle('Cumulative Output Sums with Different Δ Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{img_root}/output_sum_comparison_with_y.png')
    # plt.show()

def main2():
    img_root = 'continous_signal_images'
    delta = 0.05

    # Define and plot an exponential input signal
    signal_func = lambda t: np.where(t < 0, 0, np.exp(-t))
    input_signal = ContinousSignal(signal_func, INF=3)
    input_signal.plot(title='Exponential Input Signal', saveTo=f'{img_root}/exponential_input_signal.png')

    # Define and plot an impulse response (unit step function)
    unit_step_func = lambda t: np.where(t >= 0, 1, 0)
    impulse_response = ContinousSignal(unit_step_func, INF=3)
    impulse_response.plot(title='Impulse Response (Unit Step)', saveTo=f'{img_root}/unit_step_impulse_response.png')

    
    plot_continous_signals(input_signal, impulse_response)

    
    plot_linear_sums(input_signal, impulse_response)
    plot_output_sum(input_signal, impulse_response)


main2()
