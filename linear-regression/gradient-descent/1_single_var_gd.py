# implementation of gradient_descent take one var only

import numpy as np

def gradient_descent(f_deriv, inital_x, step_size = 0.001, precision = 0.00001):
    cur_x = inital_x                     # initial start
    last_x = float('inf')
    x_list = [cur_x]             # let's maintain our x movements

    while abs(cur_x - last_x) > precision:
        last_x = cur_x

        gradient = f_deriv(cur_x)
        cur_x -= gradient * step_size   # move in opposite direction

        x_list.append(cur_x)            # keep copy of what we visit

    print(f'The minimum y exists at x {cur_x}')
    return x_list

def trial1():
    def f(x):
        return 3 * x ** 2 + 4 * x + 7       # 3x² + 4x + 7


    def f_derivative(x):
        return 6 * x + 4                    # derivative of f(x)

    func_name = 'Gradient Descent on 3x^2 + 4x + 7'

    print(func_name)

    for inital_x in [-7.5, 5, -2/3]:
        gradient_descent(f_derivative, inital_x, step_size = 0.01)

    '''
    The minimum y exists at x -0.6668200405024292
    The minimum y exists at x -0.666513535786557
    The minimum y exists at x -0.6666666666666666
    '''

def trial2():
    def f(x):
        return x ** 4 - 6 * x ** 2 - x - 1  # x⁴ - 6x² - x -1

    def f_derivative(x):
        return 4 * x ** 3 - 12 * x - 1      # 4x³ − 12x − 1


    func_name = 'Gradient Descent on x⁴ - 6x² - x - 1'

    print(func_name)
    
    for inital_x in [-2.4, -0.15, 0.1, 2.39]:
        gradient_descent(f_derivative, inital_x, step_size = 0.001)
        '''
        The minimum y exists at x -1.6892154238200836
        The minimum y exists at x -1.6883429437023425
        The minimum y exists at x 1.7719305891118124
        The minimum y exists at x 1.7726747523515027
        '''

if __name__ == '__main__':
    trial1()
    trial2()
