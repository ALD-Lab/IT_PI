# IT_PI: Information-theoretic Buckingham-Π theorem
_The code implements IT-PI, an information-theoretic, data-driven method to identify the most predictive dimensionless inputs for non-dimensional quantities._
## Introduction
Dimensional analysis is one of the most fundamental tools for
understanding physical systems. However, the construction of
dimensionless variables, as guided by the Buckingham-$\pi$ theorem, is
not uniquely determined. 
Here, we introduce IT-$\pi$, a model-free
method that combines dimensionless learning with the principles of
information theory. Grounded in the irreducible error theorem,
IT-$\pi$ identifies dimensionless variables with the highest
predictive power by measuring their shared information content.
## Features 
- IT-PI identifies optimal dimensionless variables.
- IT-PI estimates optimal error bounds and its uncertainty based on discovered dimensionless variables.
- IT-PI distinguishes physical regimes.
- IT-PI discovers self-similar variables.
- IT-PI extracts characteristic scales.
## Getting started
Use the following command to install all required libraries:
```sh
pip install numpy scipy matplotlib pandas cma
```

Clone the repository, then import the module:
```python
import IT_PI
```

Here is a quick start tutorial for using the module. 
```python
import IT_PI
from scipy.special import erf
import numpy as np

# Define Rayleigh velocity profile
def velocity_profile(y, nu, U, t):
    return U * (1 - erf(y / (2 * np.sqrt(nu * t))))

# Generate synthetic dataset
y_vals = np.linspace(0, 1, 20)
t_vals = np.linspace(5, 10, 20)
U = np.random.uniform(0.5, 1.0, 5)
nu = np.random.uniform(1e-3, 1e-2, 5)

u_array = []
params = []

for u0, n0 in zip(U, nu):
    for y in y_vals:
        for t in t_vals:
            u_array.append(velocity_profile(y, n0, u0, t))
            params.append([u0, y, t, n0])

u_array = np.array(u_array).reshape(-1, 1)
params = np.array(params)

# Define inputs and outputs
Y = u_array / params[:, 0].reshape(-1, 1)  # Output Pi_o = u/U
X = params                                 # Dimensional input list q
variables = ['U', 'y', 't', '\\nu']        # Variable names
D_in = np.matrix('1 1 0 2; -1 0 1 -1')     # Dimension matrix
num_input = 1

# Print input shape
print("Input shape:", X.shape)

# Run dimensionless learning
results = IT_PI.main(
    X,
    Y,
    D_in,
    variables=variables,
    num_input=num_input,
    estimator="kraskov",
    estimator_params={"k": 5},
    seed=42
)

# Display results
print("\nOptimal dimensionless variable(s):", results["labels"])
print("Irreducible error:", results["irreducible_error"])
print("Uncertainty:", results["uncertainty"])
```
## Example Notebooks
See the Jupyter notebooks in `Examples` directory for results presented in the paper. This includes:

**Five validation cases:**

• **Rayleigh** — Rayleigh problem  
• **Colebrook** — Colebrook equation  
• **Malkus** — Malkus waterwheel  
• **Benard** — Rayleigh–Bénard convection  
• **Blasius** — Blasius boundary layer  


**Five application cases:**

• **Velocity_transformation** — Velocity scaling for compressible wall-bounded turbulence  
• **Wall_model** — Wall fluxes in compressible turbulence over rough walls  
• **Dixit** — Wall friction in turbulent flow over smooth surfaces under different mean pressure gradients  
• **Keyhole** — Formation of a keyhole in a puddle of liquid metal melted by a laser  
• **MHD** — Magnetohydrodynamics power generator 


