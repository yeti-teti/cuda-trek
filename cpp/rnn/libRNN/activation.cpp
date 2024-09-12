
#include <iostream.h>
#include <math.h>

using namespace std;

int tanh(int x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
