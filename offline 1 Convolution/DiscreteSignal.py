import numpy as np 
import matplotlib.pyplot as plt


class DiscreteSignal:
    def __init__(self, INF):
        self.INF = INF
        self.signal = np.zeros (2 * INF + 1)    

    def set_value_at_time(self, time , value):
        self.signal[self.INF + time ]= value 

    def shift_signal(self, shift):
       
        new_signal = np.roll(self.signal, shift)
    
        new_signal_obj = DiscreteSignal(self.INF)

        if shift > 0:  
            new_signal[:shift] = 0
        elif shift < 0: 
            new_signal[shift:] = 0  

        new_signal_obj.signal = new_signal
        return new_signal_obj

        
    def add (self, other):
        new_signal = self.signal + other.signal
        new_signal_obj = DiscreteSignal(self.INF)  
        new_signal_obj.signal = new_signal  
        return new_signal_obj
    
    def multiply (self, other):
        new_signal = self.signal * other.signal
        new_signal_obj = DiscreteSignal(self.INF)  
        new_signal_obj.signal = new_signal  
        return new_signal_obj
    
    def multiply_const_factor (self, scalar):
        new_signal = self.signal * scalar
        new_signal_obj = DiscreteSignal(self.INF)  
        new_signal_obj.signal = new_signal  
        return new_signal_obj
    
    def plot(self, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None):
        plt.figure(figsize=figsize)
        plt.xticks(np.arange(-self.INF, self.INF + 1, 1))
        y_range = (y_range[0], max(np.max(self.signal), y_range[1]) + 1)

        plt.stem(np.arange(-self.INF, self.INF+1), self.signal)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        if saveTo is not None:
            plt.savefig(saveTo)
        plt.show()


#EXAMPLE
# img_root = '.'
# signal1 = DiscreteSignal(10)
# signal1.set_value_at_time(0,0.5)
# signal1.set_value_at_time(1, 2)
# signal1.set_value_at_time(3, 1)
# signal1.set_value_at_time(5, 3)
# signal1.set_value_at_time(-1, -1)
# signal1.set_value_at_time(-3, 2)
# signal1.set_value_at_time(-5, 1)

# signal2 = signal1.shift_signal(3)

# signal1.plot(title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
# signal2.plot(title='Shifted Signal(x[n-3])', saveTo=f'{img_root}/x[n-3].png')

# signal1.set_value_at_time(0, 1)
# signal2 = signal1.shift_signal(3)
# signal3 = signal1.add(signal2)
# signal4 = signal3.multiply(signal2)
# signal3.plot(title='Original Signal(x[n]) + Shifted Signal(x[n-3])', saveTo=f'{img_root}/x[n] + x[n-3].png')
# signal4.plot(title='Original Signal(x[n]) * Shifted Signal(x[n-3])', saveTo=f'{img_root}/x[n] multi x[n-3].png')

# signal2.plot( title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
