import numpy as np
import matplotlib.pyplot as plt
from ContinousSignal import ContinousSignal

class LTI_Continous:
    def __init__(self,impulse_response):
        self.impulse_response = impulse_response
    
    def linear_combination_of_impulses(self, input_signal,delta):
        unit_impulses = []
        coefficients = []
        # input_signal.plot(title='Input Signal', saveTo='input_signal.png')
        sum_signal = ContinousSignal(lambda t: 0, input_signal.INF)
        
        for i in range (int (-input_signal.INF/delta),int (input_signal.INF/delta)):
            print (i)
            unit_impulse_func = lambda t: np.where((t >0) & (t <= delta), 1/delta, 0)
        
            unit_impulse = ContinousSignal(unit_impulse_func,input_signal.INF)
            # unit_impulse.plot(title=f'Unit Impulse at t={i*delta}', saveTo=f'unit_impulse_at_t_{i*delta}.png')
            new_signal = unit_impulse.shift(i*delta).multiply_const_factor((input_signal.evaluate(i*delta)*delta))
            # new_signal.plot(title=f'&(t-({i}∇))x({i}∇)∇', saveTo=f'unit_impulse_at_t_{i*delta}.png')
            # sum_signal = sum_signal.add(new_signal)

            # if (i == 1 or i== 2):
            #     new_signal.plot(title=f'&(t-({i}∇))x({i}∇)∇', saveTo=f'{img_root}/unit_impulse_at_t_{i}.png')
            # sum_signal.plot(title='Sum of Impulses', saveTo=f'{img_root}/each_of{i}_impulses.png')
            unit_impulses.append(new_signal)
            coefficients.append(input_signal.evaluate(i*delta))
        print(coefficients)
        # sum_signal.plot(title='Sum of Impulses', saveTo='sum_of_impulses.png')
        # input_signal.plot(title='Input Signal', saveTo='input_signal.png')
        return unit_impulses, coefficients
    
    def output_approx (self, input_signal,delta):
        unit_impulses, coefficients = self.linear_combination_of_impulses(input_signal,delta)
        output_unit_impulses = []   
        output_signal = ContinousSignal(lambda t: 0, input_signal.INF)
        for i in range (int(-input_signal.INF/delta),int (input_signal.INF/delta)):
            new_signal = self.impulse_response.shift(i*delta).multiply_const_factor(coefficients[i + int(input_signal.INF/delta)]*delta)
            output_unit_impulses.append(new_signal)
            output_signal = output_signal.add(new_signal)
            # new_signal.plot(title=f'h(t-({i}∇))x({i}∇)∇', saveTo=f'{img_root}/output_signal_at_t_{i*delta}.png')
        # output_signal.plot(title=f'Output Signal{delta}', saveTo=f'output_signal{delta}.png')
        return output_unit_impulses





#test case  

# img_root = 'continous_signal_images'

# signal_func = lambda t:np.where(t < 0, 0, np.exp(-t))
# signal1 = ContinousSignal(signal_func,3)
# delta = 0.05
# unit_step_functions = lambda t: np.where(t >= 0, 1, 0)
# lti = LTI_Continous(ContinousSignal(unit_step_functions,3))
# unit_impulses, coefficients = lti.linear_combination_of_impulses(signal1,delta)


# sum_signal = ContinousSignal(lambda t: 0, signal1.INF)
# for i in range(len(unit_impulses)):
#     # unit_impulses[i].plot(title=f'&(t-({i}∇))x({i}∇)∇', saveTo=f'{img_root}/unit_impulse_at_t_{i}.png')
#     sum_signal = sum_signal.add(unit_impulses[i])

# sum_signal.plot(title='Sum of Impulses', saveTo=f'{img_root}/sum_of_impulses.png')    
# # output_signal = lti.output_approx(signal1,delta)