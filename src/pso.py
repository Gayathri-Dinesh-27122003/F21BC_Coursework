# PSO Implementation based on Algorithm 39 from "Essentials of Metaheuristics" Section 3.5
# This implementation uses the informants topology variant
#Used ChatGPT to write the comments for this file.

import numpy as np
import random

class Particle:
    
    def __init__(self, dimension, bounds=None):
        
        self.dimension = dimension
        self.bounds = bounds
        
        # Algorithm 39, Line 2: Initialize particle position randomly
        if bounds:
            self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        else:
            self.position = np.random.uniform(-1, 1, dimension)
        
        # Algorithm 39, Line 3: Initialize particle velocity randomly
        self.velocity = np.random.uniform(-1, 1, dimension) * 0.1
        # Algorithm 39, Line 4: Initialize personal best position
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        # Informants list for topology-based communication
        self.informants = []
    
    def update_velocity(self, w, c1, c2, global_best_position):
        """Update particle velocity using PSO velocity equation."""
        # Algorithm 39, Line 9: Generate random numbers r1, r2
        r1 = random.random()
        r2 = random.random()
        # Algorithm 39, Line 10: Calculate cognitive component (personal best attraction)
        cognitive = c1 * r1 * (self.best_position - self.position)
        # Algorithm 39, Line 10: Calculate social component (neighborhood best attraction)
        social = c2 * r2 * (global_best_position - self.position)
        # Algorithm 39, Line 10: Update velocity: v = w*v + c1*r1*(p_best - x) + c2*r2*(n_best - x)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, bounds=None):
        """Update particle position based on velocity."""
        # Algorithm 39, Line 11: Update position: x = x + v
        self.position = self.position + self.velocity
        # Boundary handling: clamp position to search space bounds
        if bounds:
            self.position = np.clip(self.position, bounds[0], bounds[1])
    
    def evaluate(self, fitness_function):
        """Evaluate the particle's fitness and update personal best."""
        # Algorithm 39, Line 5: Evaluate fitness at current position
        fitness = fitness_function(self.position)
        # Algorithm 39, Line 6-7: Update personal best if current fitness is better
        if fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness
        return fitness


class PSO:
    # Implementation of Algorithm 39 (PSO with informants) from "Essentials of Metaheuristics"
    # Parameters w, c1, c2 follow standard PSO constriction coefficient values
    
    def __init__(self, objective_function, dimension, 
                 swarm_size=30, num_informants=3,
                 w=0.729, c1=1.49445, c2=1.49445,
                 bounds=None, max_iterations=100):
        
        self.objective_function = objective_function
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.num_informants = num_informants  # Number of informants per particle (topology)
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient
        self.c2 = c2    # Social coefficient
        self.bounds = bounds
        self.max_iterations = max_iterations
        
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        
        # Algorithm 39, Line 1: Initialize swarm
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize the particle swarm."""
        # Algorithm 39, Line 1-4: Create and initialize all particles
        self.swarm = []
        for i in range(self.swarm_size):
            particle = Particle(self.dimension, self.bounds)
            self.swarm.append(particle)
        # Initialize informants topology (neighborhood structure)
        self._update_informants()
    
    def _update_informants(self):
        """Assign informants to each particle."""
        # Algorithm 39 variant: Informants topology for neighborhood structure
        # Each particle is influenced by a subset of other particles (informants)
        for particle in self.swarm:
            possible_informants = [p for p in self.swarm if p is not particle]
            particle.informants = random.sample(possible_informants, 
                                              min(self.num_informants, len(possible_informants)))
    
    def _get_informants_best(self, particle):
        """Get the best position among a particle's informants."""
        # Algorithm 39, Line 8: Find neighborhood best (n_best)
        # In informants topology, neighborhood = particle's informants
        best_fitness = float('inf')
        best_position = None
        
        for informant in particle.informants:
            if informant.best_fitness < best_fitness:
                best_fitness = informant.best_fitness
                best_position = informant.best_position
        
        if best_position is None:
            return self.global_best_position
        return best_position
    
    def optimize(self, verbose=True):
        # Main optimization loop following Algorithm 39 structure
        if verbose:
            print(f" Starting PSO Optimization")
            print(f"   Swarm size: {self.swarm_size}")
            print(f"   Informants per particle: {self.num_informants}")
            print(f"   Search dimension: {self.dimension}")
            print(f"   Max iterations: {self.max_iterations}")
        
        # Algorithm 39: Main iteration loop
        for iteration in range(self.max_iterations):
            # Algorithm 39, Lines 5-7: Evaluate fitness and update personal bests
            for particle in self.swarm:
                fitness = particle.evaluate(self.objective_function)
                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness
            
            # Periodically update informants topology (dynamic neighborhood)
            if iteration % 5 == 0:
                self._update_informants()
            
            # Algorithm 39, Lines 8-11: Update velocities and positions
            for particle in self.swarm:
                # Line 8: Get neighborhood best from informants
                informants_best = self._get_informants_best(particle)
                # Line 10: Update velocity
                particle.update_velocity(self.w, self.c1, self.c2, informants_best)
                # Line 11: Update position
                particle.update_position(self.bounds)
            
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"   Iteration {iteration:3d}: Best Fitness = {self.global_best_fitness:.6f}")
        
        if verbose:
            print(f" Optimization completed!")
            print(f"   Final best fitness: {self.global_best_fitness:.6f}")
            print(f"   Total iterations: {self.max_iterations}")
        
        return self.global_best_position, self.global_best_fitness, self.fitness_history
    
    def get_optimization_summary(self):
        """Get summary of optimization results."""
        return {
            'best_fitness': self.global_best_fitness,
            'best_position_shape': self.global_best_position.shape,
            'total_iterations': self.max_iterations,
            'swarm_size': self.swarm_size,
            'num_informants': self.num_informants,
            'fitness_history': self.fitness_history
        }
