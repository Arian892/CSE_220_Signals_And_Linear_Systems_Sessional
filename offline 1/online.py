import numpy as np
from LTI_Discrete import LTI_Discrete
from DiscreteSignal import DiscreteSignal

# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split()))
n = int(input("Window size: "))

length = price_list.__len__()
sum = n * (n + 1)/ 2
print (sum )



unweighted_impulse = DiscreteSignal(length-1)
weighted_impulse  = DiscreteSignal(length-1)

for i in range (n):
    unweighted_impulse.set_value_at_time(i,1/n)
    weighted_impulse.set_value_at_time(i,(i+1)/sum)

signal_a = DiscreteSignal(length-1)
for i in range (length):
    signal_a.set_value_at_time(i,price_list[i])

# signal_a.plot()  
# unweighted_impulse.plot()
# weighted_impulse.plot()  


unwet=LTI_Discrete(unweighted_impulse)
wet = LTI_Discrete(weighted_impulse)

arr_un,sum_un =unwet.output(signal_a)
arr_w ,sum_w= wet.output(signal_a)

print (sum_un.signal)
print (sum_w.signal)

sum_un.plot()
sum_w.plot()





# price_list = [1, 2, 3, 4, 5, 6, 7, 8]
# n = 4

# Please determine uma and wma.

# Unweighted Moving Averages as a Python list
uma = []

# Weighted Moving Averages as a Python list
wma = []

# Print the two moving averages
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))