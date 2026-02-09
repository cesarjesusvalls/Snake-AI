# NeuroSnake

Neuroevolution of neural networks to play Snake using a genetic algorithm. Brains start with random weights and, over generations of selection, crossover, and mutation, learn to navigate the board, avoid walls/themselves, and eat food.

<p align="center">
  <img src="demo.gif" alt="NeuroSnake demo" width="400">
</p>

## How It Works

### Neural Network (SnakeBrain)

Each snake is controlled by a small feedforward neural network:

- **Input (12 features)**:
  - 3 values for food direction (straight, left, right)
  - 3 values for death checks (wall or body one step away: straight, left, right)
  - 3 values for tail direction (straight, left, right)
  - 3 values for flood fill reachable space ratio (straight, left, right)
- **Hidden layer**: 16 neurons with ReLU activation
- **Output**: 3 actions (turn left, turn right, go straight) via softmax

All directions are relative to the snake's heading, so the network reasons about "left of me" and "right of me" rather than compass directions. The flood fill inputs detect dead-end pockets before entering them by running BFS from each candidate position.

### Genetic Algorithm

1. **Initialize** a population of N brains with random weights
2. **Evaluate** each brain over 20 game trials, averaging fitness
3. **Rank** the population by fitness
4. **Select** the top performers as elites (preserved unchanged)
5. **Reproduce** by picking two parents (exponential selection bias toward top performers), crossing over their weights, and applying Gaussian mutation
6. **Repeat** for many generations

| Parameter         | Default |
|-------------------|---------|
| Population size   | 100     |
| Elite size        | 10      |
| Mutation rate     | 15%     |
| Mutation strength | 0.1     |
| Trials per eval   | 20      |
| Generations       | 100     |

### Fitness

Fitness = food score + efficiency bonus. The efficiency bonus rewards snakes that find food in fewer steps, normalized by the starvation limit (150 steps).

### Game Rules

- 20x20 grid board
- Snake dies on wall collision or self-collision
- Game ends after 150 steps without eating food (starvation)
- Score = number of food items eaten

## Installation

**Requirements**: Python 3.9+

```bash
# Clone the repo
git clone https://github.com/cjesus/SnakeAI.git
cd SnakeAI

# Install as a package (creates the `neurosnake` CLI command)
pip install .

# Or install in editable/development mode
pip install -e .
```

Alternatively, install dependencies manually and run the script directly:

```bash
pip install -r requirements.txt
python neurosnake.py
```

## Usage

```bash
# Run evolution (installed)
neurosnake

# Or run directly
python neurosnake.py
```

This evolves a population of 100 snakes over 100 generations. Every 5 generations (after generation 15), the best game so far is replayed in a Pygame window. Generation statistics are printed to the console:

```
Generation 42 Statistics:
Best Score: 12.40
Average Score: 3.25
Worst Score: 0.00
```

After evolution completes, results are saved to a timestamped folder under `./output/`:

```
output/run_20260208_122626/
├── best_brain.pt     # PyTorch state_dict of the best neural network
└── best_game.json    # Recorded game of the best snake
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--generations N` | Number of generations to run | 100 |
| `--population N` | Population size | 100 |
| `--workers N` | Worker processes for parallel evaluation | CPU count |
| `--no-viz` | Disable visualization | off |
| `--show-every N` | Replay best game every N generations | 5 |
| `--show-after N` | Start replaying after generation N | 15 |
| `--only-new` | Only replay when the best score improves | off |
| `--output-dir DIR` | Directory to save results | `./output` |
| `--seed-brain PATH` | Seed initial population from a saved brain (clones + mutations) | - |
| `--load-brain PATH` | Load a saved brain and play live games | - |
| `--num-games N` | Number of games to play with loaded brain | 1 |
| `--replay PATH` | Replay a saved `best_game.json` recording | - |
| `--gif PATH` | Export a `best_game.json` recording as an animated GIF | - |
| `--gif-output PATH` | Output path for GIF (default: same directory as input) | - |
| `--gif-speed MS` | Frame duration in milliseconds for GIF | 80 |

### Examples

```bash
# Run evolution with larger population
neurosnake --generations 200 --population 200

# Headless run with specific worker count
neurosnake --no-viz --generations 50 --workers 8

# Seed from a previously trained brain
neurosnake --seed-brain output/run_.../best_brain.pt --generations 100

# Replay a saved best game
neurosnake --replay output/run_.../best_game.json

# Load a trained brain and watch it play live
neurosnake --load-brain output/run_.../best_brain.pt --num-games 5

# Export a game recording as GIF
neurosnake --gif output/run_.../best_game.json
neurosnake --gif output/run_.../best_game.json --gif-output demo.gif --gif-speed 60
```

## Project Structure

```
SnakeAI/
├── neurosnake.py      # Game engine, neural network, genetic algorithm, visualizer
├── pyproject.toml     # Package configuration
├── requirements.txt   # Dependencies (for manual install)
├── LICENSE            # MIT license
└── README.md
```

Brain evaluation is parallelized across CPU cores using `ProcessPoolExecutor`.

## Dependencies

| Package | Version     | Purpose                        |
|---------|-------------|--------------------------------|
| pygame  | >= 2.5.0    | Game visualization & rendering |
| torch   | >= 2.0.0    | Neural network implementation  |
| numpy   | >= 1.24, <2 | Numerical operations           |
| Pillow  | >= 9.0.0    | GIF export                     |

## License

MIT
