import numpy as np
import matplotlib.pyplot as plt
from DiscreteSignal import DiscreteSignal


class LTI_Discrete:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response
        self.INF = impulse_response.INF 
    

    def linear_combination_of_impulses(self, input_signal):
        unit_impulses = []
        coefficients = []
        # input_signal.plot(title='Input Signal', saveTo='input_signal.png')
        # input_signal.shift_signal(1).plot(title='Shifted Input Signal', saveTo='shifted_input_signal.png')

        for i in range(-input_signal.INF, input_signal.INF + 1):
            if input_signal.signal[i + input_signal.INF] != 0:
                new_signal = DiscreteSignal(input_signal.INF)
                new_signal.set_value_at_time(i, 1)
            else :

                new_signal = DiscreteSignal(input_signal.INF)
                
            new_signal = new_signal.multiply(input_signal)
            # title = f'&[n-({i})]x[{i}]'
            # new_signal.plot(title=title, saveTo=f'{img_root}/unit_impulse_at_n_{i}.png')
            unit_impulses.append(new_signal)    

            coefficients.append(float(input_signal.signal[input_signal.INF + i]))
     
        print(coefficients)
        return unit_impulses, coefficients 
    def output (self, input_signal):
        unit_impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        output_signal = DiscreteSignal(input_signal.INF)
        output_unit_impulses = []
        

        for i in range (-input_signal.INF, input_signal.INF + 1):
            new_signal = DiscreteSignal(input_signal.INF)
            new_signal = self.impulse_response.shift_signal(i)
            new_signal = new_signal.multiply_const_factor(coefficients[i + input_signal.INF])
            output_unit_impulses.append(new_signal)
            output_signal = output_signal.add(new_signal)
            # title = f'h[n-({i})]x[{i}]'
            # new_signal.plot(title=title, saveTo=f'{img_root}/output_signal_at_n_{i}.png')

      
            #  output_signal.plot(title='Output Signal', saveTo=f'{img_root}/output_signal.png')
            print(output_signal)
        return output_unit_impulses,output_signal
#test case

# img_root = 'signal_images'
# signal1 = DiscreteSignal(5)
# signal1.set_value_at_time(0,0.5)
# signal1.set_value_at_time(1, 2)

# signal2 = DiscreteSignal(5)
# signal2.set_value_at_time(0, 1)
# signal2.set_value_at_time(1, 1)
# signal2.set_value_at_time(2, 1)

# lti = LTI_Discrete(signal2)
# lti.output(signal1)
