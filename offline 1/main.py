import numpy as np
import matplotlib.pyplot as plt
from DiscreteSignal import DiscreteSignal
from LTI_Discrete import LTI_Discrete

def plot_discrete_signals(input_signal, impulse_response_signal):
    img_root = 'discrete_signal_images'

    # Create an LTI system with the given impulse response
    lti = LTI_Discrete(impulse_response_signal)

    # Get unit impulses from linear combination without applying coefficients
    unit_impulses, _ = lti.linear_combination_of_impulses(input_signal)
    
    # Calculate cumulative sum of unit impulses directly
    cumulative_sum_impulses = DiscreteSignal(input_signal.INF)
    for impulse in unit_impulses:
        cumulative_sum_impulses = cumulative_sum_impulses.add(impulse)
    
    # Plot unit impulses and their cumulative sum
    fig1, axs1 = plt.subplots((len(unit_impulses) + 1) // 3, 3, figsize=(15, 3 * ((len(unit_impulses) + 1) // 3)))
    axs1 = axs1.flatten()
    
    for idx, impulse in enumerate(unit_impulses):
      
        time_index = idx - impulse.INF
        
        axs1[idx].stem(np.arange(-impulse.INF, impulse.INF + 1), impulse.signal, basefmt=" ")
        axs1[idx].set_title(f'&[n-({time_index})]x[{time_index}]')
        axs1[idx].set_xlabel('n (Time Index)')
        axs1[idx].set_ylabel('x[n]')
        axs1[idx].grid(True)
    

    axs1[len(unit_impulses)].stem(np.arange(-cumulative_sum_impulses.INF, cumulative_sum_impulses.INF + 1), cumulative_sum_impulses.signal, basefmt=" ")
    axs1[len(unit_impulses)].set_title('Cumulative Sum (Impulses)')
    axs1[len(unit_impulses)].set_xlabel('n (Time Index)')
    axs1[len(unit_impulses)].set_ylabel('x[n]')
    axs1[len(unit_impulses)].grid(True)

 
    for idx in range(len(unit_impulses) + 1, len(axs1)):
        axs1[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{img_root}/linear_combination_impulses.png')
    plt.show()


    output_unit_impulses,_ = lti.output(input_signal)
    cumulative_output_sum = DiscreteSignal(input_signal.INF)
    for output_component in output_unit_impulses:
        cumulative_output_sum = cumulative_output_sum.add(output_component)
    

    fig2, axs2 = plt.subplots((len(output_unit_impulses) + 1) // 3, 3, figsize=(15, 3 * ((len(output_unit_impulses) + 1) // 3)))
    axs2 = axs2.flatten()
    
    for idx, component in enumerate(output_unit_impulses):
        
        time_index = idx - component.INF
       
        axs2[idx].stem(np.arange(-component.INF, component.INF + 1), component.signal, basefmt=" ")
        axs2[idx].set_title(f'h[n-({time_index})]x[{time_index}]')
        axs2[idx].set_xlabel('n (Time Index)')
        axs2[idx].set_ylabel('x[n]')
        axs2[idx].grid(True)

 
    axs2[len(output_unit_impulses)].stem(np.arange(-cumulative_output_sum.INF, cumulative_output_sum.INF + 1), cumulative_output_sum.signal, basefmt=" ")
    axs2[len(output_unit_impulses)].set_title('Cumulative Sum (Output)')
    axs2[len(output_unit_impulses)].set_xlabel('n (Time Index)')
    axs2[len(output_unit_impulses)].set_ylabel('x[n]')
    axs2[len(output_unit_impulses)].grid(True)

    
    for idx in range(len(output_unit_impulses) + 1, len(axs2)):
        axs2[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{img_root}/output_components.png')
    plt.show()

def main():
    
    discrete_signal = DiscreteSignal(5)
    discrete_signal.set_value_at_time(0, 0.5)
    discrete_signal.set_value_at_time(1, 2)

   
    impulse_response = DiscreteSignal(5)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(2, 1)

    
    plot_discrete_signals(discrete_signal, impulse_response)


main()
