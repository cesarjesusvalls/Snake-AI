# SnakeAI

A neuroevolution project that evolves neural networks to play Snake using a genetic algorithm. Brains start with random weights and, over generations of selection, crossover, and mutation, learn to navigate the board, avoid walls/themselves, and eat food.

## How It Works

### Neural Network (SnakeBrain)

Each snake is controlled by a small feedforward neural network built with PyTorch:

- **Input (28 features)**:
  - 3 values for relative food direction (straight, left, right)
  - 25 values from a 5x5 "death awareness" grid centered on the snake's head (walls and body = danger)
- **Hidden layer**: 5 neurons with ReLU activation
- **Output**: 3 actions (turn left, turn right, go straight) via softmax

The network uses relative directions rather than absolute ones, so the snake reasons about "left of me" and "right of me" rather than compass directions.

### Genetic Algorithm

The `GeneticAlgorithm` class evolves a population of brains:

1. **Initialize** a population of N brains with random weights
2. **Evaluate** each brain over 20 game trials, averaging scores
3. **Rank** the population by fitness (food eaten)
4. **Select** the top performers as elites (preserved unchanged)
5. **Reproduce** by picking two parents (exponential selection bias toward top performers), crossing over their weights, and applying Gaussian mutation
6. **Repeat** for many generations

| Parameter         | Default |
|-------------------|---------|
| Population size   | 50      |
| Elite size        | 10      |
| Mutation rate     | 15%     |
| Mutation strength | 0.1-0.4 |
| Trials per eval   | 20      |
| Generations       | 100     |

### Game Rules

- 20x20 grid board
- Snake dies on wall collision or self-collision
- Game ends after 75 steps without eating food (starvation)
- Score = number of food items eaten

## Project Structure

```
SnakeAI/
├── genetics.py        # Game engine, neural network, genetic algorithm, visualizer
└── requirements.txt   # Python dependencies
```

Brain evaluation is parallelized across CPU cores using `ProcessPoolExecutor`.

## Setup

**Requirements**: Python 3.9+, conda (recommended)

```bash
# Create and activate a conda environment
conda create -n snakeai python=3.11 -y
conda activate snakeai

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python genetics.py
```

This evolves a population of 50 snakes over 100 generations. Every 5 generations (after generation 15), the best game so far is replayed in a Pygame window. Generation statistics are printed to the console:

```
Generation 12 Statistics:
Best Score: 4.20
Average Score: 1.05
Worst Score: 0.00
```

After evolution completes, results are saved to a timestamped folder under `./output/`:

```
output/run_20260208_102921/
├── best_brain.pt     # PyTorch state_dict of the best neural network
└── best_game.json    # Recorded game of the best snake
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--generations N` | Number of generations to run | 100 |
| `--population N` | Population size | 50 |
| `--workers N` | Worker processes for parallel evaluation | CPU count |
| `--no-viz` | Disable visualization | off |
| `--show-every N` | Replay best game every N generations | 5 |
| `--show-after N` | Start replaying after generation N | 15 |
| `--only-new` | Only replay when the best score improves | off |
| `--output-dir DIR` | Directory to save results | `./output` |
| `--load-brain PATH` | Load a saved brain and play live games | - |
| `--num-games N` | Number of games to play with loaded brain | 1 |
| `--replay PATH` | Replay a saved best game recording | - |

### Examples

```bash
# Run evolution
python genetics.py --generations 200 --population 100

# Run evolution with specific worker count
python genetics.py --generations 200 --workers 8

# Headless run
python genetics.py --no-viz --generations 50

# Replay a saved best game
python genetics.py --replay output/run_.../best_game.json

# Load a trained brain and watch it play live
python genetics.py --load-brain output/run_.../best_brain.pt --num-games 5
```

## Dependencies

| Package | Version     | Purpose                        |
|---------|-------------|--------------------------------|
| pygame  | >= 2.5.0    | Game visualization & rendering |
| torch   | >= 2.0.0    | Neural network implementation  |
| numpy   | >= 1.24, <2 | Numerical operations           |

> **Note**: numpy must be < 2.0 due to a breaking change in `np.where` scalar conversion behavior.
