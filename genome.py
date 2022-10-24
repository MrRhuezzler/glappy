from audioop import cross
from os import stat
import re
from select import select
from typing import List
import random
import numpy as np
from brain import FCLayer, sigmoid

class Genome:
    def __init__(self, network: List[FCLayer]) -> None:
        self.network = network
        self.fitness = 0

class Genetics:
    @staticmethod
    def __create_child(g: Genome, genes):
        layers = []
        for weight in Genetics.__unflatten_gene(g, genes):
            intermediate_layer = FCLayer(
                weight.shape[0], weight.shape[1], sigmoid)
            intermediate_layer.set(weight)
            layers.append(intermediate_layer)
        return Genome(layers)

    @staticmethod
    def __flatten_gene(g:  Genome):
        genes = []
        for layer in g.network:
            genes.extend(layer.flatten())
        return genes

    @staticmethod
    def __unflatten_gene(g: Genome, genes):
        weights = []
        start = 0
        for layer in g.network:
            size = layer.input_size * layer.output_size
            weight = np.array(
                genes[start: start + size]).reshape((layer.input_size, layer.output_size))
            weights.append(weight)
            start += size
        return weights

    @staticmethod
    def cross(g1: Genome, g2: Genome):
        g1_genes = Genetics.__flatten_gene(g1)
        g2_genes = Genetics.__flatten_gene(g2)

        length = len(g1_genes)
        split = random.randint(0, length - 1)

        child1_genes = g1_genes[:split] + g2_genes[split:]
        child2_genes = g1_genes[split:] + g2_genes[:split]

        child1 = Genetics.__create_child(g1, child1_genes)
        child2 = Genetics.__create_child(g1, child2_genes)

        return child1, child2
    
    @staticmethod
    def selection(genomes: List[Genome]):
        genomes = sorted(genomes, key=lambda x: x.fitness, reverse=True)
        return genomes[:int(0.2 * len(genomes))]


    @staticmethod
    def crossover(genomes: List[Genome], pop_size):
        offsprings = []
        for _ in range(pop_size - len(genomes) // 2):
            p1 = random.choice(genomes)
            p2 = random.choice(genomes)
            c1, c2 = Genetics.cross(p1, p2)
            offsprings.append(c1)
            offsprings.append(c2)
        genomes.extend(offsprings)
        return genomes[:pop_size]

    @staticmethod
    def mutation(genomes: List[Genome]):
        for genome in genomes:
            if random.uniform(0.0, 1.0) <= 0.1:
                genes = Genetics.__flatten_gene(genome)
                randint = random.randint(0,len(genes)-1)
                genes[randint] = np.random.randn()
                weights = Genetics.__unflatten_gene(genome, genes)
                for weight, layer in zip(weights, genome.network):
                    layer.set(weight)        
        return genomes

    @staticmethod
    def evolve(fit_function, genomes: List[Genome], gen, pop_size):
        fit_function(genomes, gen, pop_size)
        pop_size = len(genomes)
        genomes = Genetics.selection(genomes)
        genomes = Genetics.crossover(genomes, pop_size)
        genomes = Genetics.mutation(genomes)
        for genome in genomes:
            genome.fitness = 0
        return genomes

    @staticmethod
    def generate_population(pop_size, network_config):
        genomes = []
        for _ in range(pop_size):
            layers = []
            for layer in network_config:
                layers.append(FCLayer(*layer))
            genomes.append(Genome(layers))
        return genomes
