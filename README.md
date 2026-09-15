# Gradient Descent: Convergence and Step-Size Experiments

An executable notebook for studying first-order optimization. It derives the
gradient-descent update, uses automatic differentiation to evaluate gradients,
and visualizes how initialization and learning rate change an optimization
trajectory.

## What is inside

- Analytical motivation for the update \(w_{t+1}=w_t-\alpha\nabla g(w_t)\)
- Gradients computed with `autograd`
- Reusable iterative optimization code
- Experiments on smooth and non-smooth objectives
- Plots that make stable, slow, and divergent step sizes easy to compare

## Quick start

```bash
git clone https://github.com/takakhoo/Gradient-Descent-Experiments.git
cd Gradient-Descent-Experiments
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab "Gradient Descent Exp.ipynb"
```

[Open the executed notebook](Gradient%20Descent%20Exp.ipynb)

## Suggested experiments

Change one variable at a time: the initial point, step size, iteration budget,
or objective. A useful extension is to compare fixed-step gradient descent with
momentum, RMSProp, or an adaptive line search under identical stopping rules.

## Scope

This is an educational optimization lab. It favors readable cells and plots
over a packaged solver API.
