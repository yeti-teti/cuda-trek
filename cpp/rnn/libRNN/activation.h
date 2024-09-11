#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <math.h>

int tanh(int x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }

#endif
