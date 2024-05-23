# Evolutionary Algorithms for TSP

## Overview
This repository contains an implementation of an evolutionary algorithm to solve the Traveling Salesman Problem (TSP). The algorithm uses selection, crossover, and mutation operations to evolve a population of solutions over multiple generations, aiming to find the shortest possible route that visits each city exactly once and returns to the origin city.

## Features
- **TSP File Reading**: Parses .tsp files to extract city coordinates.
- **Distance Matrix Creation**: Computes the pairwise distances between all cities.
- **Fitness Function**: Evaluates the fitness of a route based on the total travel distance.
- **Crossover and Mutation**: Implements genetic operators to generate new solutions.
- **Parent Selection**: Selects parents using tournament selection.
- **Evolutionary Algorithm**: Combines the above components to evolve a population of routes over multiple generations.

## Files
- **`evolutionary_algorithm.py`**: Contains the main implementation of the evolutionary algorithm and helper functions.
- **`.tsp files`**: Contains several .tsp files with city coordinates used as input data for the algorithm.
  - `city1.tsp`
  - `city2.tsp`
  - `city3.tsp`
  - `zi929.tsp`

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or higher
- Libraries: numpy

You can install the required package using the following command:
```bash
pip install numpy
