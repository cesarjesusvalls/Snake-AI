import copy
import numpy as np
from typing import List, Tuple
import torch.nn as nn
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Action(Enum):
    LEFT = 'L'   # Turn left relative to current direction
    RIGHT = 'R'  # Turn right relative to current direction
    NOTHING = 'N'  # Continue in current direction

@dataclass
class GameState:
    score: int
    steps: int
    snake_positions: List[Tuple[int, int]]
    food_position: Tuple[int, int]
    current_direction: Tuple[int, int]
    steps_without_food: int

@dataclass
class GameRecord:
    initial_position: Tuple[int, int]
    initial_direction: Tuple[int, int]
    actions: List[Action]
    food_positions: List[Tuple[int, int]]
    final_score: int
    total_steps: int

class SnakeGame:
    def __init__(self, board_size: int = 20, cell_size: int = 20):
        self.board_size = board_size
        self.cell_size = cell_size
        self.window_size = board_size * cell_size
        
        # Game state
        self.reset()
        
        # Direction mappings for relative turns
        self.direction_map = {
            Action.LEFT: self._turn_left,
            Action.RIGHT: self._turn_right,
            Action.NOTHING: lambda direction: direction
        }

    def reset(self, initial_position: Optional[Tuple[int, int]] = None,
             initial_direction: Optional[Tuple[int, int]] = None):
        """Reset the game state."""
        if initial_position is None:
            initial_position = (self.board_size // 2, self.board_size // 2)
        if initial_direction is None:
            initial_direction = (1, 0)  # Start moving right
            
        self.snake_positions = [initial_position]
        self.current_direction = initial_direction
        self.food_position = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.game_over = False
        self.growth_remaining = 0
        
        # Additional debug info
        self.hit_wall = False
        self.self_collision = False

    def _spawn_food(self, position: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Spawn food at given position or random position if None."""
        if position is not None:
            return position
            
        available_positions = [
            (x, y) for x in range(self.board_size) for y in range(self.board_size)
            if (x, y) not in self.snake_positions
        ]
        return random.choice(available_positions) if available_positions else (0, 0)

    def _turn_left(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        """Turn 90 degrees left relative to current direction."""
        return (-direction[1], direction[0])
    
    def _turn_right(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        """Turn 90 degrees right relative to current direction."""
        return (direction[1], -direction[0])

    def get_state(self) -> GameState:
        """Return current game state."""
        return GameState(
            score=self.score,
            steps=self.steps,
            snake_positions=self.snake_positions.copy(),
            food_position=self.food_position,
            current_direction=self.current_direction,
            steps_without_food=self.steps_without_food
        )

    def get_occupancy_map(self) -> np.ndarray:
        """Return the current occupancy map (for AI input)."""
        occupancy = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Mark snake body
        for x, y in self.snake_positions[:-1]:
            occupancy[y, x] = 1
        
        # Mark snake head with 2
        head_x, head_y = self.snake_positions[-1]
        occupancy[head_y, head_x] = 2
        
        # Mark food with 3
        food_x, food_y = self.food_position
        occupancy[food_y, food_x] = 3
        
        return occupancy

    def step(self, action: Action) -> bool:
        """Execute one game step. Returns False if game is over."""
        if self.game_over:
            return False

        # Update direction based on action (using relative turning)
        self.current_direction = self.direction_map[action](self.current_direction)

        # Calculate new head position
        head_x, head_y = self.snake_positions[-1]
        new_x = head_x + self.current_direction[0]
        new_y = head_y + self.current_direction[1]
        
        # Check wall collision
        if new_x < 0 or new_x >= self.board_size or new_y < 0 or new_y >= self.board_size:
            self.game_over = True
            self.hit_wall = True
            return False

        new_head = (new_x, new_y)

        # Check collision with self
        if new_head in self.snake_positions:
            self.game_over = True
            self.self_collision = True
            return False

        # Move snake
        self.snake_positions.append(new_head)
        
        # Check if food is eaten
        if new_head == self.food_position:
            self.score += 1
            self.steps_without_food = 0
            self.growth_remaining += 1
            self.food_position = self._spawn_food()
        
        # Handle snake growth
        if self.growth_remaining > 0:
            self.growth_remaining -= 1
        else:
            self.snake_positions.pop(0)
            self.steps_without_food += 1

        # Check if snake is stuck
        if self.steps_without_food >= 75:
            self.game_over = True
            return False

        self.steps += 1
        return True

class SnakeBrain(nn.Module):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size
        
        # Input features:
        # - 3 for food direction (straight, left, right)
        # - 25 for death awareness (5x5 grid centered on snake head)
        input_size = 3 + 25
        
        # Enhanced network architecture for handling more inputs
        self.fc1 = nn.Linear(input_size, 5)
        #self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(5, 3)  # 3 outputs for LEFT, RIGHT, NOTHING
        
        # Initialize with random weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1, 1)
    
    def _get_death_matrix(self, occupancy_map: np.ndarray, head_pos: Tuple[int, int]) -> np.ndarray:
        """Get a 5x5 death awareness matrix centered on the snake's head."""
        # Create a 5x5 matrix filled with death (1) by default
        death_matrix = np.ones((5, 5), dtype=float)
        head_x, head_y = head_pos
        
        # Iterate through the 5x5 area centered on the head
        for dy in range(-2, 3):  # -2, -1, 0, 1, 2
            for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                x = head_x + dx
                y = head_y + dy
                
                # Check if position is within board boundaries
                if (0 <= x < self.board_size and 0 <= y < self.board_size):
                    # Mark as safe (0) if the position is empty or contains food
                    if occupancy_map[y, x] == 0 or occupancy_map[y, x] == 3:
                        death_matrix[dy + 2, dx + 2] = 0
        
        return death_matrix
    
    def _get_relative_food_direction(self, head_pos: Tuple[int, int], food_pos: Tuple[int, int], 
                                   current_direction: Tuple[int, int]) -> Tuple[bool, bool, bool]:
        """
        Determine if food is straight ahead, to the left, or to the right relative to current direction.
        Returns tuple of (straight, left, right).
        """
        dx = food_pos[0] - head_pos[0]
        dy = food_pos[1] - head_pos[1]
        
        # Convert current direction to angle
        current_angle = np.arctan2(current_direction[1], current_direction[0])
        # Convert food direction to angle
        food_angle = np.arctan2(dy, dx)
        
        # Calculate relative angle
        angle_diff = (food_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Check directions
        straight = abs(angle_diff) < np.pi/4
        left = angle_diff > np.pi/4 and angle_diff < 3*np.pi/4
        right = angle_diff < -np.pi/4 and angle_diff > -3*np.pi/4
        
        return straight, left, right
    
    def _get_input_features(self, occupancy_map: np.ndarray, current_direction: Tuple[int, int]) -> torch.Tensor:
        """Get input features for the neural network."""
        # Find head position
        head_pos = tuple(map(int, np.where(occupancy_map == 2)))[::-1]  # Reverse to get (x, y)
        
        # Find food position
        food_pos = tuple(map(int, np.where(occupancy_map == 3)))[::-1]  # Reverse to get (x, y)
        
        # Get food direction relative to snake's current direction
        food_straight, food_left, food_right = self._get_relative_food_direction(head_pos, food_pos, current_direction)
        
        # Get death awareness matrix
        death_matrix = self._get_death_matrix(occupancy_map, head_pos)
        
        # Create input tensor
        # First 3 values are food direction
        inputs = [float(food_straight), float(food_left), float(food_right)]
        
        # Add flattened death matrix (25 values)
        inputs.extend(death_matrix.flatten().tolist())
        
        return torch.tensor(inputs, dtype=torch.float32)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
    def get_action(self, occupancy_map: np.ndarray, current_direction: Tuple[int, int]) -> Action:
        """Convert network output to snake action."""
        # Get input features
        input_features = self._get_input_features(occupancy_map, current_direction)
        
        # Add batch dimension
        x = input_features.unsqueeze(0)
        
        # Get network prediction
        with torch.no_grad():
            action_probs = self(x)
        
        # Convert to numpy and get action index
        action_idx = action_probs.numpy().argmax()
        
        # Map index to action
        actions = [Action.LEFT, Action.RIGHT, Action.NOTHING]
        return actions[action_idx]


class SnakeVisualizer:
    def __init__(self, cell_size: int = 20):
        self.cell_size = cell_size
        self.font = None
        self.screen = None
        self.colors = {
            'background': (0, 0, 0),
            'snake': (0, 255, 0),
            'food': (255, 0, 0),
            'head': (0, 200, 0),
            'text': (255, 255, 255)
        }
        self.board_size = None

    def initialize(self):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 16)

    def setup_display(self, board_size: int):
        self.initialize()
        self.board_size = board_size
        self.window_size = board_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
        pygame.display.set_caption('Snake Game')

    def cleanup(self):
        pygame.quit()
        self.screen = None
        self.font = None

    def draw_state(self, state: GameState):
        if not self.screen:
            self.setup_display(self.board_size)
            
        self.screen.fill(self.colors['background'])
        
        # Draw snake body
        for x, y in state.snake_positions[:-1]:
            pygame.draw.rect(self.screen, self.colors['snake'],
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Draw snake head
        head_x, head_y = state.snake_positions[-1]
        pygame.draw.rect(self.screen, self.colors['head'],
                        (head_x * self.cell_size, head_y * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw food
        food_x, food_y = state.food_position
        pygame.draw.rect(self.screen, self.colors['food'],
                        (food_x * self.cell_size, food_y * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw grid
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (0, i * self.cell_size), 
                           (self.window_size, i * self.cell_size))
        
        pygame.display.flip()

    def clear_info_area(self):
        if not self.screen:
            return
        pygame.draw.rect(self.screen, self.colors['background'],
                        (self.window_size, 0, 200, self.window_size))

    def draw_info(self, text: str):
        if not self.screen or not self.font:
            return
        
        self.clear_info_area()
            
        x = self.window_size + 10
        y = 10
        for line in text.split('\n'):
            text_surface = self.font.render(line, True, self.colors['text'])
            self.screen.blit(text_surface, (x, y))
            y += 20
        pygame.display.flip()

    def replay_game(self, game_record: GameRecord, speed_ms: int = 100):
        """Replay a recorded game with fixed food positioning."""
        try:
            self.setup_display(20)  # Use standard board size
            
            game = SnakeGame(board_size=self.board_size, cell_size=self.cell_size)
            game.reset(game_record.initial_position, game_record.initial_direction)
            
            # Initialize with first food position
            food_index = 0
            game.food_position = game_record.food_positions[food_index]
            
            clock = pygame.time.Clock()
            running = True
            action_index = 0
            
            while running and action_index < len(game_record.actions):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                
                if not running:
                    break
                
                # Get the next action from the record
                action = game_record.actions[action_index]
                
                # Get current state before step for visualization
                current_state = game.get_state()
                
                # Visualize current state and info
                self.draw_state(current_state)
                self.draw_info(
                    f"Replay - Step: {action_index + 1}/{len(game_record.actions)}\n"
                    f"Score: {game.score}\n"
                    f"Action: {action.value}\n"
                    f"Direction: {game.current_direction}\n"
                    f"Head: {game.snake_positions[-1]}\n"
                    f"Food: {game.food_position}"
                )
                
                # Perform the action
                if game.step(action):
                    # Check if food was eaten
                    if len(game.snake_positions) > len(current_state.snake_positions):
                        food_index += 1
                        if food_index < len(game_record.food_positions):
                            game.food_position = game_record.food_positions[food_index]
                    
                    action_index += 1
                    clock.tick(1000 / speed_ms)
                else:
                    # Game over state
                    self.draw_state(game.get_state())
                    self.draw_info(
                        f"GAME OVER!\n"
                        f"Final Score: {game.score}\n"
                        f"Total Steps: {action_index}\n"
                        f"Hit Wall: {game.hit_wall}\n"
                        f"Self Collision: {game.self_collision}\n"
                        f"Steps without food: {game.steps_without_food}"
                    )
                    pygame.display.flip()
                    time.sleep(2)
                    break
                
        finally:
            self.cleanup()

def simulate_game(brain: SnakeBrain, game: SnakeGame, visualizer: Optional[SnakeVisualizer] = None, 
                 speed_ms: int = 100) -> GameRecord:
    """Simulate a full game and record all states and actions."""
    actions = []
    food_positions = [game.food_position]
    initial_position = game.snake_positions[0]
    initial_direction = game.current_direction
    
    clock = pygame.time.Clock() if visualizer else None
    
    while not game.game_over:
        occupancy = game.get_occupancy_map()
        action = brain.get_action(occupancy, game.current_direction)
        actions.append(action)
        
        if visualizer:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            
            visualizer.draw_state(game.get_state())
            visualizer.draw_info(
                f"Score: {game.score}  Steps: {len(actions)}\n"
                f"Direction: {game.current_direction}\n"
                f"Action: {action.value}\n"
                f"Steps without food: {game.steps_without_food}"
            )
            clock.tick(1000 / speed_ms)
        
        # Store food position before step
        prev_food = game.food_position
        
        if not game.step(action):
            if visualizer:
                visualizer.draw_state(game.get_state())
                visualizer.draw_info(
                    f"GAME OVER!\n"
                    f"Final Score: {game.score}  Steps: {len(actions)}\n"
                    f"Hit wall: {game.hit_wall}\n"
                    f"Self collision: {game.self_collision}\n"
                    f"Stuck: {game.steps_without_food >= 50}"
                )
                pygame.display.flip()
                time.sleep(2)
            break
        
        # Record new food position if it changed
        if game.food_position != prev_food:
            food_positions.append(game.food_position)
    
    return GameRecord(
        initial_position=initial_position,
        initial_direction=initial_direction,
        actions=actions,
        food_positions=food_positions,
        final_score=game.score,
        total_steps=len(actions)
    )



@dataclass
class SnakeGenetics:
    generation: int
    scores: List[float]
    best_score: float
    average_score: float
    worst_score: float
    best_game: GameRecord

class GeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 elite_size: int = 10,
                 mutation_rate: float = 0.15,
                 mutation_strength: float = 0.4):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        
    def _evaluate_brain(self, brain: SnakeBrain, num_trials: int = 20, visualizer: Optional[SnakeVisualizer] = None) -> Tuple[float, Optional[GameRecord]]:
        """Evaluate a single brain by running multiple games and averaging the scores."""
        total_score = 0
        best_score = float('-inf')
        best_record = None
        game = SnakeGame(board_size=brain.board_size)
        
        for trial in range(num_trials):
            game.reset()
            # Use visualizer only on last trial if it's provided
            current_visualizer = visualizer if trial == num_trials - 1 else None
            game_record = simulate_game(brain, game, current_visualizer)
            #print(game_record.final_score)
            score = game_record.final_score#(game_record.final_score+1)*game_record.total_steps
            
            total_score += score
            
            # Keep track of best performance
            if score > best_score:
                best_score = score
                best_record = game_record
            
        return total_score / num_trials, best_record
    def _extract_weights(self, brain: SnakeBrain) -> List[np.ndarray]:
        """Extract weights from a brain as numpy arrays."""
        return [p.data.numpy().copy() for p in brain.parameters()]  # Added .copy() for deep copy

    def _inject_weights(self, brain: SnakeBrain, weights: List[np.ndarray]):
        """Inject weights into a brain."""
        for param, weight in zip(brain.parameters(), weights):
            param.data = torch.from_numpy(weight.copy())  # Added .copy() for deep copy

    def _crossover(self, parent1_weights: List[np.ndarray], 
                  parent2_weights: List[np.ndarray]) -> List[np.ndarray]:
        """Perform crossover between two parents' weights."""
        child_weights = []
        for w1, w2 in zip(parent1_weights, parent2_weights):
            mask = np.random.rand(*w1.shape) < 0.5
            child_w = np.where(mask, w1, w2).copy()  # Added .copy() for deep copy
            child_weights.append(child_w)
        return child_weights

    def _mutate(self, weights: List[np.ndarray]):
        """Apply mutation to weights."""
        for w in weights:
            mutation_mask = np.random.rand(*w.shape) < self.mutation_rate
            mutations = np.random.normal(0, self.mutation_strength, w.shape)
            w[mutation_mask] += mutations[mutation_mask]

    def evolve_population(self, board_size: int, visualizer: Optional[SnakeVisualizer] = None) -> SnakeGenetics:
        """Run one generation of evolution."""
        # Initialize population if first generation
        if self.generation == 0:
            self.population = [SnakeBrain(board_size) for _ in range(self.population_size)]
            
        # Evaluate all brains
        print(f"\nEvaluating Generation {self.generation}")
        scores = []
        game_records = []
        
        # Evaluate each brain
        for i, brain in enumerate(self.population):
            print(f"Evaluating Snake {i+1}/{self.population_size}", end='\r')
            
            # Run evaluation trials
            score, game_record = self._evaluate_brain(brain, visualizer=None)
            scores.append(score)
            game_records.append(game_record)
        
        # Sort population, scores, and game records together
        sorted_data = sorted(zip(scores, self.population, game_records), 
                           key=lambda x: x[0], reverse=True)
        scores, self.population, game_records = zip(*sorted_data)
        scores = list(scores)
        self.population = list(self.population)
        game_records = list(game_records)
        
        # Now visualize the actual best performing snake
        if visualizer:
            print("\nReplicating best snake's performance...")
            best_score, best_record = self._evaluate_brain(
                self.population[0],  # Best snake after sorting
                num_trials=1,  # Single trial for visualization
                visualizer=visualizer
            )
            game_records[0] = best_record  # Update the record with the visualized performance
        
        # Create next generation
        new_population = []
        
        # Keep elite individuals
        new_population.extend(copy.deepcopy(self.population[:self.elite_size]))
        
        # Create rest of new population
        while len(new_population) < self.population_size:
            # Select parents (bias towards better performing individuals)
            parent1_idx = int(np.random.exponential(scale=self.population_size/25))
            parent2_idx = int(np.random.exponential(scale=self.population_size/25))
            parent1_idx = min(parent1_idx, len(self.population) - 1)
            parent2_idx = min(parent2_idx, len(self.population) - 1)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Create child through crossover
            child = SnakeBrain(board_size)
            child_weights = self._crossover(
                self._extract_weights(parent1),
                self._extract_weights(parent2)
            )
            
            # Apply mutation
            self._mutate(child_weights)
            
            # Inject weights into child
            self._inject_weights(child, child_weights)
            
            new_population.append(child)
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Return statistics
        return SnakeGenetics(
            generation=self.generation - 1,
            scores=scores,
            best_score=max(scores),
            average_score=sum(scores) / len(scores),
            worst_score=min(scores),
            best_game=game_records[0]  # Record of the best snake
        )


def main(show_visualization: bool = True):
    # Parameters
    board_size = 20
    num_generations = 100
    population_size = 50
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(population_size=population_size)
    
    # Create visualizer only if visualization is enabled
    visualizer = None
    if show_visualization:
        visualizer = SnakeVisualizer(cell_size=30)
    
    # Keep track of best game ever
    best_game_ever = None
    best_score_ever = float('-inf')
    
    try:
        # Evolution loop
        for gen in range(num_generations):
            # Evolve population (evaluate without visualization)
            stats = ga.evolve_population(board_size, None)
            
            # Check if this is the best game ever
            if stats.best_score > best_score_ever:
                best_score_ever = stats.best_score
                best_game_ever = stats.best_game
            
            # Print statistics
            print(f"\nGeneration {stats.generation} Statistics:")
            print(f"Best Score: {stats.best_score:.2f}")
            print(f"Average Score: {stats.average_score:.2f}")
            print(f"Worst Score: {stats.worst_score:.2f}")
            
            # Show replay of the best game if visualization is enabled
            if show_visualization and best_game_ever and gen%5==0 and gen>15:
                print("Replaying best game ever...")
                #print(best_game_ever)
                #print(best_game_ever)
                visualizer.replay_game(best_game_ever, speed_ms=50)
                time.sleep(1)  # Pause between generations
            
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
    finally:
        if visualizer:
            visualizer.cleanup()

if __name__ == "__main__":
    main(show_visualization=False)


