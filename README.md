# IT_PI: Information-theoretic Buckingham-Π theorem
_The code implements IT-PI, an information-theoretic, data-driven framework inspired by Buckingham PI theorem, which leverages mutual information based bounds to identify the most predictive dimensionless inputs for non-dimensional quantities._
## Introduction
Physical laws and models must rely on dimensionless variables. The Buckingham PI theorem systematically derives dimensionless numbers, though they are not unique and optimal for prediction.
We introduce IT-PI, an information-theoretic, data-driven framework inspired by Buckingham PI theorem, which leverages mutual information based bounds to identify the most predictive dimensionless inputs for non-dimensional quantities.
Grounded in the information-theoretic bounds to the irreducible model error, IT-PI maximizes predictability of the output regardless of the chosen modeling approach. 
The method involves the maximization of the mutual information between inputs and output, which is efficiently solved using the covariance matrix adaptation evolution strategy algorithm. 
IT-PI applies to algebraic, ODE, and PDE relationships, identifies optimal dimensionless inputs, distinguishes physical regimes, discovers self-similar variables, extracts characteristic scales, and provides optimal model bounds based on discovered dimensionless variables.
