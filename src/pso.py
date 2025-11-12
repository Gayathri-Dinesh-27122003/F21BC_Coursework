import numpy as np
import random

class Particle:
    """
    Represents a single particle in the PSO swarm.
    Each particle represents a candidate solution (ANN weights).
    """
    
    def __init__(self, dimension, bounds=None):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            dimension (int): Dimensionality of the search space (number of ANN parameters)
            bounds (tuple): Optional bounds for particle positions (min, max)
        """
        self.dimension = dimension
        self.bounds = bounds
        
        if bounds:
            self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        else:
            self.position = np.random.uniform(-1, 1, dimension)
        
        self.velocity = np.random.uniform(-1, 1, dimension) * 0.1
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.informants = []
    
    def update_velocity(self, w, c1, c2, global_best_position):
        """Update particle velocity using PSO velocity equation."""
        r1 = random.random()
        r2 = random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, bounds=None):
        """Update particle position based on velocity."""
        self.position = self.position + self.velocity
        if bounds:
            self.position = np.clip(self.position, bounds[0], bounds[1])
    
    def evaluate(self, fitness_function):
        """Evaluate the particle's fitness and update personal best if improved."""
        fitness = fitness_function(self.position)
        if fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness
        return fitness


class PSO:
    """
    Particle Swarm Optimization implementation with informants (Algorithm 39).
    """
    
    def __init__(self, objective_function, dimension, 
                 swarm_size=30, num_informants=3,
                 w=0.729, c1=1.49445, c2=1.49445,
                 bounds=None, max_iterations=100):
        """
        Initialize PSO optimizer.
        
        Args:
            objective_function (callable): Function to minimize
            dimension (int): Dimensionality of search space (ANN parameter count)
            swarm_size (int): Number of particles in swarm
            num_informants (int): Number of informants per particle
            w (float): Inertia weight
            c1 (float): Cognitive acceleration coefficient
            c2 (float): Social acceleration coefficient
            bounds (tuple): Optional bounds for particle positions
            max_iterations (int): Maximum number of iterations
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.num_informants = num_informants
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.max_iterations = max_iterations
        
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize the particle swarm."""
        self.swarm = []
        for i in range(self.swarm_size):
            particle = Particle(self.dimension, self.bounds)
            self.swarm.append(particle)
        self._update_informants()
    
    def _update_informants(self):
        """Assign informants to each particle."""
        for particle in self.swarm:
            possible_informants = [p for p in self.swarm if p is not particle]
            particle.informants = random.sample(possible_informants, 
                                              min(self.num_informants, len(possible_informants)))
    
    def _get_informants_best(self, particle):
        """Get the best position among a particle's informants."""
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
        """
        Run the PSO optimization.
        
        Returns:
            tuple: (best_position, best_fitness, history)
        """
        if verbose:
            print(f"🚀 Starting PSO Optimization")
            print(f"   Swarm size: {self.swarm_size}")
            print(f"   Informants per particle: {self.num_informants}")
            print(f"   Search dimension: {self.dimension}")
            print(f"   Max iterations: {self.max_iterations}")
            print("=" * 50)
        
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                fitness = particle.evaluate(self.objective_function)
                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness
            
            if iteration % 5 == 0:
                self._update_informants()
            
            for particle in self.swarm:
                informants_best = self._get_informants_best(particle)
                particle.update_velocity(self.w, self.c1, self.c2, informants_best)
                particle.update_position(self.bounds)
            
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"   Iteration {iteration:3d}: Best Fitness = {self.global_best_fitness:.6f}")
        
        if verbose:
            print("=" * 50)
            print(f"✅ Optimization completed!")
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
