import pandas as pd
from pygad import GA
from sklearn.neural_network import MLPClassifier

from constants import SEED, LAYER_SIZES, ACTIVATIONS, SOLVERS, LEARNING_RATES
from models.model import Model


class NN(Model):
    def __init__(self, x: pd.DataFrame, y: pd.Series, search_space: list):
        """
        Initializes the NN class.
        This class serves as a base for Neural Network models.
        """
        super().__init__(MLPClassifier, x, y)
        self.search_space = search_space
        self.ga = GA(
            fitness_func=self.fitness_function,
            num_genes=len(self.search_space),
            gene_space=self.search_space,
            num_generations=10,
            num_parents_mating=5,
            sol_per_pop=10,
            mutation_probability=0.2,
            random_seed=SEED
        )

    def run(self):
        """
        Runs the genetic algorithm to optimize the model parameters.
        """
        self.ga.run()

    def fitness_function(self, ga_instance, solution, solution_idx):
        params = {
            'hidden_layer_sizes': LAYER_SIZES[int(solution[0])],
            'activation': ACTIVATIONS[int(solution[1])],
            'solver': SOLVERS[int(solution[2])],
            'learning_rate': LEARNING_RATES[int(solution[3])],
            'learning_rate_init': solution[4],
            'alpha': solution[5],
            'max_iter': int(solution[6]),
        }

        fitness = self.train_val_split(params)
        print(f"({solution_idx})\n\tEvaluated parameters: {params}\n\tAccuracy: {fitness:.4f}")

        params['accuracy'] = fitness
        self.solutions.append(params)

        return fitness