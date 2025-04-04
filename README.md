
## 🛠️ Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

Install with:

```bash
pip install -r requirements.txt

# 🧬 Rastrigin_RS – Genetic Algorithm Optimization (Lab 3)

This project implements a Genetic Algorithm (GA) in Python to minimize the 2D Rastrigin function using **Rank Selection** as the parent selection strategy. It was developed as part of **Artificial Intelligence Lab 3** (Variant 2) for educational purposes.

## 📌 Objective

To find the **global minimum** of the Rastrigin function, a classic benchmark in evolutionary optimization, by evolving a population of real-valued solutions using biologically-inspired operators.

## ⚙️ Algorithm Overview

- **Selection**: Rank-based (individuals selected with probabilities based on their fitness ranking)
- **Crossover**: Random interpolation using real-valued parents  
  \[
  x_o = \alpha x_{p1} + (1 - \alpha) x_{p2}
  \]
- **Mutation**: Gaussian mutation applied to a random portion of the offspring
- **Elitism**: Part of the population is preserved unaltered

## 📈 Target Function

\[
f(x, y) = 20 + (x^2 - 10\cos(2\pi x)) + (y^2 - 10\cos(2\pi y))
\]

- **Search domain**: \( x, y \in [-5, 5] \)
- **Global minimum**: \( f(0, 0) = 0 \)

## 🧪 Experiments

The code supports 4 experiments:

1. **Hyperparameter search** (grid search over population size, mutation rate/strength, crossover rate, generations)
2. **Randomness & population impact** (multiple seeds + reduced population sizes)
3. **Crossover impact** (plots average/best fitness for different crossover rates)
4. **Mutation impact** (plots convergence behavior for multiple mutation rates & strengths)

All results (plots + CSVs) are saved in the `/data_test/` folder.

## 📂 Project Structure

