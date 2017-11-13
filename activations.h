//========================================================================
// Activation header file
//========================================================================
// @brief: function prototype & activate type definition

#ifndef SRC_ACTIVATIONS_H_
#define SRC_ACTIVATIONS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// activation type
typedef enum
{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

// get activation type
ACTIVATION get_activation(char *s);
// activation
float activate(float x, ACTIVATION a);
// activation in batch mode
void activate_array(float *x, const int n, const ACTIVATION a);
// add gradient
float gradient(float x, ACTIVATION a);
// add gradient in batch mode
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);

//activation helper function
static inline float logistic_activate(float x)
{
    return 1.0/(1.0 + exp(-x));
}
static inline float logistic_gradient(float x)
{
    return (1-x)*x;
}
static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0)
    {
        return floor(x/2.0);
    }
    else
    {
        return (x - n) + floor(x/2.0);
    }
}
static inline float hardtan_activate(float x)
{
    if (x < -1)
    {
        return -1;
    }
    if (x > 1)
    {
        return 1;
    }
    return x;
}
static inline float linear_activate(float x)
{
    return x;
}
static inline float loggy_activate(float x)
{
    return 2.0/(1.0 + exp(-x)) - 1;
}
static inline float relu_activate(float x)
{
    return x*(x>0);
}
static inline float elu_activate(float x)
{
    return (x >= 0)*x + (x < 0)*(exp(x)-1);
}
static inline float relie_activate(float x)
{
    return (x>0) ? x : 0.01*x;
}
static inline float ramp_activate(float x)
{
    return x*(x>0)+0.1*x;
}
static inline float leaky_activate(float x)
{
    return (x>0) ? x : 0.1*x;
}
static inline float tanh_activate(float x)
{
    return (exp(2*x)-1)/(exp(2*x)+1);
}
static inline float plse_activate(float x)
{
    if(x < -4)
    {
        return 0.01 * (x + 4);
    }
    if(x > 4)
    {
        return 0.01 * (x - 4) + 1;
    }
    return 0.125*x + .5;
}
static inline float lhtan_activate(float x)
{
    if(x < 0)
    {
        return 0.001*x;
    }
    if(x > 1)
    {
        return 0.001*(x-1) + 1;
    }
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1)
    {
        return 1;
    }
    return 0.001;
}
static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1)
    {
        return 1;
    }
    return 0;
}
static inline float linear_gradient(float x)
{
    return 1;
}
static inline float loggy_gradient(float x)
{
    float y = (x+1.0)/2.0;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x)
    {
        return 0;
    }
    return 1;
}
static inline float relu_gradient(float x)
{
    return (x>0);
}
static inline float elu_gradient(float x)
{
    return (x >= 0) + (x < 0)*(x + 1);
}
static inline float relie_gradient(float x)
{
    return (x>0) ? 1 : 0.01;
}
static inline float ramp_gradient(float x)
{
    return (x>0)+0.1;
}
static inline float leaky_gradient(float x)
{
    return (x>0) ? 1 : 0.1;
}
static inline float tanh_gradient(float x)
{
    return 1-x*x;
}
static inline float plse_gradient(float x)
{
    return (x < 0 || x > 1) ? 0.01 : 0.125;
}

#endif /* SRC_ACTIVATIONS_H_ */
