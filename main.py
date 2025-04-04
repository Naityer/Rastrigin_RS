import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from functions import rastrigin_2d


if __name__ == "__main__":
    # Crear carpeta de salida si no existe
    output_dir = "data_test"
    os.makedirs(output_dir, exist_ok=True)

    param_grid = {
        "population_size": [50, 100],
        "mutation_rate": [0.05, 0.1],
        "mutation_strength": [0.1, 0.5],
        "crossover_rate": [0.5, 0.9],
        "num_generations": [100],
    }

    results = []
    combinations = list(itertools.product(
        param_grid["population_size"],
        param_grid["mutation_rate"],
        param_grid["mutation_strength"],
        param_grid["crossover_rate"],
        param_grid["num_generations"],
    ))

    for idx, (pop, m_rate, m_strength, c_rate, n_gen) in enumerate(combinations):
        print(f"Running combination {idx + 1}/{len(combinations)}")

        ga = GeneticAlgorithm(
            population_size=pop,
            mutation_rate=m_rate,
            mutation_strength=m_strength,
            crossover_rate=c_rate,
            num_generations=n_gen,
        )
        best_solutions, best_fitness_values, avg_fitness_values = ga.evolve(seed=42)

        results.append({
            "population": pop,
            "mutation_rate": m_rate,
            "mutation_strength": m_strength,
            "crossover_rate": c_rate,
            "generations": n_gen,
            "best_fitness": best_fitness_values[-1],
            "average_fitness": avg_fitness_values[-1],
            "solution": best_solutions[-1],
        })

        # Optional plot for each run
        plt.plot(best_fitness_values, label=f'Run {idx+1}')

    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Best Fitness Across Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Guardar gráfica en carpeta
    plot_path = os.path.join(output_dir, "fitness_plot_exp1.png")
    plt.savefig(plot_path)
    plt.show()

    # Guardar tabla de resultados en carpeta
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "experiment1_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved results to {csv_path}")
