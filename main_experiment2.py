import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

output_dir = "data_test"
os.makedirs(output_dir, exist_ok=True)

# ✅ Usa los mejores parámetros encontrados previamente
best_params = {
    "population_size": 100,
    "mutation_rate": 0.1,
    "mutation_strength": 0.5,
    "crossover_rate": 0.9,
    "num_generations": 100,
}

random_seeds = [10, 20, 30, 40, 50]
results = []

print("▶ Running with 5 different seeds...")

# 🔹 Parte 1: Diferentes seeds
for seed in random_seeds:
    ga = GeneticAlgorithm(
        population_size=best_params["population_size"],
        mutation_rate=best_params["mutation_rate"],
        mutation_strength=best_params["mutation_strength"],
        crossover_rate=best_params["crossover_rate"],
        num_generations=best_params["num_generations"],
    )
    _, best_fitness, _ = ga.evolve(seed=seed)
    results.append(best_fitness[-1])

results = np.array(results)
summary_df = pd.DataFrame({
    "Best Fitness Across Seeds": results,
})
summary_df.loc["Mean"] = summary_df.mean()
summary_df.loc["Std"] = summary_df.std()
summary_df.loc["Min"] = summary_df.min()

summary_path = os.path.join(output_dir, "experiment2_randomness_summary.csv")
summary_df.to_csv(summary_path)
print(f"✅ Saved randomness summary to {summary_path}")

# 🔹 Parte 2: Poblaciones más pequeñas
print("\n▶ Running with smaller populations...")

pop_factors = [0.5, 0.25, 0.1]
small_pop_results = []

for factor in pop_factors:
    pop_size = int(best_params["population_size"] * factor)
    run_fitnesses = []
    for seed in random_seeds:
        ga = GeneticAlgorithm(
            population_size=pop_size,
            mutation_rate=best_params["mutation_rate"],
            mutation_strength=best_params["mutation_strength"],
            crossover_rate=best_params["crossover_rate"],
            num_generations=best_params["num_generations"],
        )
        _, best_fitness, _ = ga.evolve(seed=seed)
        run_fitnesses.append(best_fitness[-1])

    small_pop_results.append({
        "Population": pop_size,
        "Mean Fitness": np.mean(run_fitnesses),
        "Std Dev": np.std(run_fitnesses),
        "Best Fitness": np.min(run_fitnesses)
    })

pop_df = pd.DataFrame(small_pop_results)
pop_path = os.path.join(output_dir, "experiment2_population_summary.csv")
pop_df.to_csv(pop_path, index=False)
print(f"✅ Saved population reduction results to {pop_path}")
