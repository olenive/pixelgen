"""Helper functions for working with neat-python."""

import neat
from typing import Dict


class NeatInterfaces:

    def set_genome_fitnesses(population: neat.Population, fitnesses: Dict[int, float]) -> None:
        """Figure out where to put this."""
        pass

    def gather_and_report_statistics(population: neat.Population) -> None:
        population.reporters.start_generation(population.generation)
        best = None
        for g in population.population.values():
            if g.fitness is None:
                raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

            if best is None or g.fitness > best.fitness:
                best = g
        population.reporters.post_evaluate(population.config, population.population, population.species, best)

    def advance_to_next_generation(population):
        """
        Use the NEAT algorithm to advance a population of genomes from the current generation to the next

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.

        Let's make a function that updates all the genomes in the population by assigning a new fitness value to them.
        """
        NeatInterfaces.gather_and_report_statistics(population)

        # Evaluate all genomes using the user-provided function.
        # fitness_function(list(self.population.items()), self.config)

        # Create the next generation from the current generation. Mutate population object.
        population.population = population.reproduction.reproduce(
            population.config, population.species, population.config.pop_size, population.generation
        )

        # Check for complete extinction.
        if not population.species.species:
            population.reporters.complete_extinction()

            # If requested by the user, create a completely new population, otherwise raise an exception.
            if population.config.reset_on_extinction:
                population.population = population.reproduction.create_new(
                    population.config.genome_type,
                    population.config.genome_config,
                    population.config.pop_size
                )
            else:
                raise neat.CompleteExtinctionException()

        # Divide the new population into species.
        population.species.speciate(population.config, population.population, population.generation)
        population.reporters.end_generation(population.config, population.population, population.species)
        population.generation += 1
