import random
import math
import numpy as np

############################################################################################
def read_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the line containing the dimension of the problem
        for line in lines:
            line = line.strip()
            if line.startswith('NODE_COORD_SECTION'):
                break
            elif line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())

        # Read the city coordinates
        for line in lines:
            line = line.strip()
            if line == 'EOF':
                break
            city_info = line.split()
            if len(city_info) == 3:
                try:
                    city_id = int(city_info[0])
                    x_coord = float(city_info[1])
                    y_coord = float(city_info[2])
                    cities.append((city_id, x_coord, y_coord))
                except ValueError:
                    continue

    # Verify the number of cities matches the specified dimension
    if len(cities) != dimension:
        raise ValueError('Mismatch between the specified dimension and the number of cities.')

    return cities
############################################################################################
def create_distance_matrix(cities):
    coordinates = np.array([(city[1], city[2]) for city in cities])
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
        distance_matrix[i] = distances
    
    return distance_matrix
############################################################################################
def fitness(route,distance_matrix):  # route: a list of route cities
    distances = distance_matrix[np.array(route[:-1]), np.array(route[1:])]
    total_distance = np.sum(distances)
    total_distance += distance_matrix[route[-1], route[0]]
    
    # to handle a special case where the total distance is very small    
    if total_distance < 1 / 1000:
        return 1000
    
    return 1 / total_distance  
############################################################################################
def crossover(parent1, parent2):
    
    size = len(parent1)

    # Randomly select two different positions for crossover
    separate_locations = sorted(random.sample(range(size), k=2))

    # Initialize child routes
    child1 = [-1] * size
    child2 = [-1] * size

    # Copy the selected portion from parents to children
    child1[separate_locations[0]:separate_locations[1] + 1] = parent1[separate_locations[0]:separate_locations[1] + 1]
    child2[separate_locations[0]:separate_locations[1] + 1] = parent2[separate_locations[0]:separate_locations[1] + 1]

    # Fill the remaining positions in children using the other parent
    fill_remaining(child1, parent2, separate_locations)
    fill_remaining(child2, parent1, separate_locations)

    return child1, child2

def fill_remaining(child, parent, separate_locations):
    
    size = len(parent)

    # Find the set of unused cities in the parent
    unused_cities = set(parent) - set(child)

    # Find the positions where the child has not been filled yet
    empty_positions = np.where(np.array(child) == -1)[0]
    num_empty = empty_positions.size

    if num_empty > 0:
        # Generate a list of remaining cities from the parent
        remaining_cities = [city for city in parent if city in unused_cities]

        # Randomly select cities to fill the empty positions
        remaining_indices = random.sample(range(len(remaining_cities)), k=num_empty)

        # Fill the empty positions with the selected cities
        for i, index in enumerate(empty_positions):
            child[index] = remaining_cities[remaining_indices[i]]

    return child
###########################################################################################
def mutate(route):
    size = len(route)
    idx1, idx2 = random.sample(range(size), k=2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route
############################################################################################
def distance_calculator(route,distance_matrix):
    
    distances = distance_matrix[np.array(route[:-1]), np.array(route[1:])]
    total_distance = np.sum(distances)
    total_distance += distance_matrix[route[-1], route[0]]
    
    return total_distance 
############################################################################################
def parent_selection(population, distance_matrix, miu, tournament_size):
    distances = np.array([fitness(individual, distance_matrix) for individual in population])
    parents = []

    for _ in range(miu):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_distances = distances[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_distances)]
        parents.append(population[winner_index])

    return parents 
############################################################################################
def save_tsp_file(solution_route, distance_matrix, filename='best_solution.tsp'):
    
    with open(filename, 'w') as file:
        
        file.write("NAME: Best Solution\n")
        file.write("TYPE: TSP\n")
        
        for i in range(len(solution_route)):
            file.write("{}  ".format( solution_route[i] ))
        
        file.write("{} ".format( solution_route[0] ))
        file.write("\n")
        file.write("\n")
        file.write("total distance for solution route: {}\n".format(distance_calculator(solution_route,distance_matrix)))
        file.write("\n")
        file.write("\n")
        file.write("EOF")
############################################################################################
def evolutionary_algorithm(miu, lam, Pc, Pm, distance_matrix, num_generations,tournament_size):
    population = []
    
    for _ in range(miu):
        individual = random.sample(range(len(distance_matrix)), len(distance_matrix))
        population.append(individual)

    #####################################################################################        
    for generation in range(num_generations):
        
        parents = parent_selection(population, distance_matrix, miu, tournament_size)

        ##################################################################
        # Crossover
        offspring = []
        
        for i in range(0, len(parents), 2):
            if random.random() < Pc:
                child1, child2 = crossover(parents[i], parents[i+1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1]])
                
        ##################################################################
        # Mutation
        for i in range(len(offspring)):
            
            if random.random() < Pm:
                
                offspring[i] = mutate(offspring[i])
                
        population.extend(offspring)
        
        ##################################################################
        population = sorted(population, key=lambda x: fitness(x,distance_matrix), reverse=True)
        population = population[:miu]

        # Print the best individual in each 10 generations
        if (generation) % 10 == 0 :
            best_individual = population[0]
            print( "Generation:", generation, "Best Individual total distance:", distance_calculator(best_individual,distance_matrix) )

    # Return the best individual found in a tsp file format
    save_tsp_file( population[0] , distance_matrix)
############################################################################################
def Main_algorithm(miu, lam, Pc, Pm, file_path , num_generations,tournament_size):
    cities = read_tsp_file(file_path)
    distance_matrix = create_distance_matrix(cities)
    evolutionary_algorithm(miu, lam, Pc, Pm, distance_matrix, num_generations, tournament_size) 