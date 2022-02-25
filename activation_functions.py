import math
from decimal import localcontext, Decimal, ROUND_HALF_UP

def sigmoid(x):


    return 1 / (1 + (math.e ** -x))

def rounded_sigmoid(x):

    # the standard python round doesn't always round op x.5
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        return int(Decimal(sigmoid(x)).to_integral_value())


def relu(x):
    return max([0, x])

def binary_threshold(x):
    return 1 if x >= 0 else 0


def dummy(x):
    return x