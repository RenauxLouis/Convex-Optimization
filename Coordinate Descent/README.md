# HW3

## Acceleration and Stochastic Gradients

Minimizing quadratic convex function with lipschitz gradient.
Testing with raw and noisy data

![Alt text](NGD.png?raw=true "Convergence of Nesterov and Gradient Descent")

## Coordinate Descent

Implementation of variants of coordinate descent:
- Stochastic with exact line search and with constant step length 1/Lmax
- Cyclic with exact line search and with constant step lengths 1/Lmax , 1/L, and 1/(sqrt(n)L)
- “Sampling without replacement” with exact and with constant step length 1/Lmax

We compare the performances on different convex quadratic problems to prove to convergence rates of the following theorems:
![Alt text](SCD.png?raw=true "Convergence of Stochastic CD")
![Alt text](CCD.png?raw=true "Convergence of Cyclic CD")