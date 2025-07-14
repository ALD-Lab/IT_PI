# IT_PI: Information-theoretic Buckingham-Π theorem
_The code implements IT-PI, an information-theoretic, data-driven framework inspired by Buckingham PI theorem, which leverages mutual information based bounds to identify the most predictive dimensionless inputs for non-dimensional quantities._
## Introduction
Physical laws and models must rely on dimensionless variables. The Buckingham PI theorem systematically derives dimensionless numbers, though they are not unique and optimal for prediction.
We introduce IT-PI, an information-theoretic, data-driven framework inspired by Buckingham PI theorem, which leverages mutual information based bounds to identify the most predictive dimensionless inputs for non-dimensional quantities.
Grounded in the information-theoretic bounds to the irreducible model error, IT-PI maximizes predictability of the output regardless of the chosen modeling approach. 
The method involves the maximization of the mutual information between inputs and output, which is efficiently solved using the covariance matrix adaptation evolution strategy algorithm. 
## Features 
- IT-PI applies to algebraic, ODE, and PDE relationships.
- IT-PI identifies optimal dimensionless inputs.
- IT-PI distinguishes physical regimes.
- IT-PI discovers self-similar variables.
- IT-PI extracts characteristic scales.
- IT-PI provides optimal model bounds based on discovered dimensionless variables.
## Getting started
Use the following command to install all required libraries:
```sh
pip install numpy scipy matplotlib pandas cma
```
See the Jupyter notebooks for examples in the paper.
