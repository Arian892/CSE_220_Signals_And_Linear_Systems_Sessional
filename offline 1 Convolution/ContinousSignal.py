import numpy as np
import matplotlib.pyplot as plt

class ContinousSignal:
    def __init__ (self,func,INF):
        self.func = func
        self.INF = INF

    def evaluate(self, t):
        return self.func(t)

    def shift (self,shift):
        new_func = lambda t: self.func(t - shift)
        return ContinousSignal(new_func,self.INF)
    
    def add (self, other):
        new_func = lambda t: self.func(t) + other.func(t)
        return ContinousSignal(new_func,self.INF)
    
    def multiply (self, other):
        new_func = lambda t: self.func(t) * other.func(t)
        return ContinousSignal(new_func,self.INF)
    
    def multiply_const_factor (self, scalar):
        new_func = lambda t: self.func(t) * scalar
        return ContinousSignal(new_func,self.INF)
    
    def plot (self,  num_points=1000, title=None, y_range=(-1, 3), figsize = (8, 5), x_label='t (Time Index)', y_label='x(t)', saveTo=None):
        t = np.linspace(-self.INF, self.INF, num_points)
        y = self.func(t)
        plt.figure(figsize=figsize)
        plt.plot(t, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        if saveTo is not None:
            plt.savefig(saveTo)
        # plt.show()




#test case

# img_root = 'Continous_signal_images'
# signal1 = ContinousSignal(lambda t: np.sin(t) , 10)
# signal2 = ContinousSignal(lambda t: np.cos(t),10)

# shifted_signal = signal1.shift(-3)
# added_signal = signal1.add(signal2)
# multiplied_signal = signal1.multiply(signal2)

# signal1.plot (title='sin(t)',  saveTo=f'{img_root}/sin_t.png')
# shifted_signal.plot (title='sin(t+3)',  saveTo=f'{img_root}/sin_t+3.png')
# multiplied_signal.plot (title='sin(t) * cos(t)', saveTo=f'{img_root}/sin_t_cos_t.png')
# added_signal.plot (title='sin(t) + cos(t)', saveTo=f'{img_root}/sin_t_plus_cos_t.png')


# signal_func = lambda t:np.where(t < 0, 0, np.exp(-t))


# piecewise_signal = ContinousSignal(signal_func,3)
# piecewise_signal = piecewise_signal.shift(9)


# piecewise_signal.plot(title='Piecewise Signal',  saveTo=f'{img_root}/piecewise_signal.png')