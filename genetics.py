"""fast_genetics.py — Optimized SnakeAI neuroevolution

Performance optimizations over genetics.py:
  1. Pure Python inference — no PyTorch/numpy in the hot loop (~9x faster forward pass)
  2. Direct position access — no occupancy map construction, no np.where scans
  3. Integer dot/cross product for food direction — no trigonometry
  4. Set-based collision detection — O(1) instead of O(n) list scan
  5. Deque-based snake body — O(1) tail removal
  6. Rejection sampling for food placement — faster for short snakes
  7. Inlined game loop in worker — eliminates method-call overhead
"""

import copy
import json
import os
import warnings
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
import random
import time

# Lazy pygame import — workers never load pygame
pygame = None

def _ensure_pygame():
    """Import pygame on demand (main process only)."""
    global pygame
    if pygame is None:
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        import pygame as _pg
        pygame = _pg


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


def save_game_record(record: GameRecord, path: str):
    """Serialize a GameRecord to JSON."""
    data = {
        "initial_position": list(record.initial_position),
        "initial_direction": list(record.initial_direction),
        "actions": [a.value for a in record.actions],
        "food_positions": [list(p) for p in record.food_positions],
        "final_score": record.final_score,
        "total_steps": record.total_steps,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_game_record(path: str) -> GameRecord:
    """Deserialize a GameRecord from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    action_map = {a.value: a for a in Action}
    return GameRecord(
        initial_position=tuple(data["initial_position"]),
        initial_direction=tuple(data["initial_direction"]),
        actions=[action_map[v] for v in data["actions"]],
        food_positions=[tuple(p) for p in data["food_positions"]],
        final_score=data["final_score"],
        total_steps=data["total_steps"],
    )


# ---------------------------------------------------------------------------
# SnakeGame — used for visualization / replay / load-brain modes only.
# The hot evaluation path uses _fast_evaluate() which inlines all game logic.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SnakeBrain — nn.Module for weight management, serialization, and viz path.
# The hot evaluation path uses raw Python lists extracted via get_fast_weights().
# ---------------------------------------------------------------------------

class SnakeBrain(nn.Module):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size

        # Input features (6 binary inputs):
        # 1. Food is straight ahead
        # 2. Food is to the left
        # 3. Food is to the right
        # 4. Death if continue straight
        # 5. Death if turn left
        # 6. Death if turn right
        input_size = 6

        # Simple network with one hidden layer
        self.fc1 = nn.Linear(input_size, 3)  # 3 outputs for LEFT, RIGHT, NOTHING

        # Initialize with random weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1, 1)

    def get_fast_weights(self):
        """Extract weights as Python lists for fast inference.

        Returns (w, b) where w is a 3x6 nested list and b is a length-3 list.
        """
        return self.fc1.weight.data.tolist(), self.fc1.bias.data.tolist()

    def _check_death(self, occupancy_map: np.ndarray, pos: Tuple[int, int], direction: Tuple[int, int]) -> bool:
        """Check if moving in the given direction from pos leads to death."""
        new_x = pos[0] + direction[0]
        new_y = pos[1] + direction[1]

        # Check wall collision
        if new_x < 0 or new_x >= self.board_size or new_y < 0 or new_y >= self.board_size:
            return True

        # Check self collision
        return occupancy_map[new_y, new_x] == 1

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
        """Get binary input features for the neural network."""
        # Find head position
        hy, hx = np.where(occupancy_map == 2)
        head_pos = (int(hx[0]), int(hy[0]))

        # Find food position
        fy, fx = np.where(occupancy_map == 3)
        food_pos = (int(fx[0]), int(fy[0]))

        # Get food direction relative to snake's current direction
        food_straight, food_left, food_right = self._get_relative_food_direction(head_pos, food_pos, current_direction)

        # Check death conditions for different moves
        # For straight ahead
        death_straight = self._check_death(occupancy_map, head_pos, current_direction)

        # For left turn
        left_dir = (-current_direction[1], current_direction[0])  # Rotate 90° left
        death_left = self._check_death(occupancy_map, head_pos, left_dir)

        # For right turn
        right_dir = (current_direction[1], -current_direction[0])  # Rotate 90° right
        death_right = self._check_death(occupancy_map, head_pos, right_dir)

        # Create input tensor
        inputs = torch.tensor([
            float(food_straight),
            float(food_left),
            float(food_right),
            float(death_straight),
            float(death_left),
            float(death_right)
        ], dtype=torch.float32)

        return inputs

    def forward(self, x):
        x = F.softmax(self.fc1(x), dim=1)
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

        # Map index to action (only 3 actions now)
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
        _ensure_pygame()
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
        _ensure_pygame()
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
    """Simulate a full game with visualization (main process only)."""
    _ensure_pygame()
    actions = []
    food_positions = [game.food_position]
    initial_position = game.snake_positions[0]
    initial_direction = game.current_direction

    clock = pygame.time.Clock()

    while not game.game_over:
        occupancy = game.get_occupancy_map()
        action = brain.get_action(occupancy, game.current_direction)
        actions.append(action)

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


# ---------------------------------------------------------------------------
# Fast evaluation — the core optimization.
# Runs the full game loop + neural network inference in pure Python.
# No PyTorch, no numpy, no occupancy maps in the hot path.
# ---------------------------------------------------------------------------

def _fast_evaluate(w, b, board_size, num_trials):
    """Evaluate a brain over multiple game trials using pure Python.

    Args:
        w: weight matrix as nested list [[w00..w05], [w10..w15], [w20..w25]]
        b: bias vector as list [b0, b1, b2]
        board_size: board dimension (e.g. 20)
        num_trials: number of games to average over

    Returns:
        (avg_score, best_game_record)
    """
    # Pre-extract all 21 weight values as locals (avoids list indexing in hot loop)
    w00, w01, w02, w03, w04, w05 = w[0]
    w10, w11, w12, w13, w14, w15 = w[1]
    w20, w21, w22, w23, w24, w25 = w[2]
    b0, b1, b2 = b

    bs = board_size
    bs_minus_1 = bs - 1
    start_pos = (bs // 2, bs // 2)
    randint = random.randint

    total_score = 0
    best_score = -1
    best_actions = None
    best_food_positions = None

    for _ in range(num_trials):
        # --- Initialize game state ---
        snake = deque([start_pos])
        snake_set = {start_pos}
        dx, dy = 1, 0  # direction: moving right

        # Spawn first food (rejection sampling)
        while True:
            fx, fy = randint(0, bs_minus_1), randint(0, bs_minus_1)
            if (fx, fy) not in snake_set:
                break
        food_x, food_y = fx, fy

        score = 0
        steps_without_food = 0
        growth = 0
        actions = []
        food_positions = [(food_x, food_y)]

        while True:
            hx, hy = snake[-1]

            # --- Food direction via dot/cross product (no trig) ---
            fdx = food_x - hx
            fdy = food_y - hy
            fwd = fdx * dx + fdy * dy       # forward component
            lx, ly = -dy, dx                 # left direction
            lft = fdx * lx + fdy * ly        # leftward component

            # Strict < matches the arctan2 boundary behavior exactly
            a_lft = abs(lft)
            a_fwd = abs(fwd)
            fs = 1.0 if fwd > 0 and a_lft < a_fwd else 0.0
            fl = 1.0 if lft > 0 and a_fwd < a_lft else 0.0
            fr = 1.0 if lft < 0 and a_fwd < a_lft else 0.0

            # --- Death checks (inline, set-based O(1) collision) ---
            # Straight
            nx, ny = hx + dx, hy + dy
            ds = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0
            # Left
            nx, ny = hx + lx, hy + ly
            dl = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0
            # Right
            rx, ry = dy, -dx
            nx, ny = hx + rx, hy + ry
            dr = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0

            # --- Forward pass (unrolled 6->3 matmul, pure Python) ---
            v0 = b0 + fs*w00 + fl*w01 + fr*w02 + ds*w03 + dl*w04 + dr*w05
            v1 = b1 + fs*w10 + fl*w11 + fr*w12 + ds*w13 + dl*w14 + dr*w15
            v2 = b2 + fs*w20 + fl*w21 + fr*w22 + ds*w23 + dl*w24 + dr*w25

            # Argmax -> action index (0=LEFT, 1=RIGHT, 2=NOTHING)
            if v0 >= v1 and v0 >= v2:
                best_i = 0
            elif v1 >= v2:
                best_i = 1
            else:
                best_i = 2

            actions.append(best_i)

            # --- Apply direction change ---
            if best_i == 0:    # LEFT
                dx, dy = -dy, dx
            elif best_i == 1:  # RIGHT
                dx, dy = dy, -dx
            # NOTHING: direction unchanged

            # --- Move ---
            new_hx = hx + dx
            new_hy = hy + dy

            # Wall collision
            if new_hx < 0 or new_hx > bs_minus_1 or new_hy < 0 or new_hy > bs_minus_1:
                break

            new_head = (new_hx, new_hy)

            # Self collision
            if new_head in snake_set:
                break

            snake.append(new_head)
            snake_set.add(new_head)

            # Food check
            if new_hx == food_x and new_hy == food_y:
                score += 1
                steps_without_food = 0
                growth += 1
                # Spawn new food (rejection sampling)
                while True:
                    fx, fy = randint(0, bs_minus_1), randint(0, bs_minus_1)
                    if (fx, fy) not in snake_set:
                        break
                food_x, food_y = fx, fy
                food_positions.append((food_x, food_y))

            # Growth / tail removal
            if growth > 0:
                growth -= 1
            else:
                removed = snake.popleft()
                snake_set.discard(removed)
                steps_without_food += 1

            # Starvation
            if steps_without_food >= 75:
                break

        total_score += score
        if score > best_score:
            best_score = score
            best_actions = actions
            best_food_positions = food_positions

    # Build GameRecord for the best game (convert int actions to Action enums)
    action_enums = [Action.LEFT, Action.RIGHT, Action.NOTHING]
    best_record = GameRecord(
        initial_position=start_pos,
        initial_direction=(1, 0),
        actions=[action_enums[i] for i in best_actions],
        food_positions=[tuple(p) for p in best_food_positions],
        final_score=best_score,
        total_steps=len(best_actions),
    )

    return total_score / num_trials, best_record


def _evaluate_brain_worker(args):
    """Worker function for parallel brain evaluation.

    Receives (w, b, board_size, num_trials) where w and b are Python lists.
    Returns (avg_score, best_game_record).
    """
    return _fast_evaluate(*args)


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
                 mutation_strength: float = 0.1,
                 num_workers: int = None):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0

        if num_workers is None:
            num_workers = min(mp.cpu_count(), population_size)
        self.num_workers = num_workers

        # Only create a process pool if using multiple workers
        if self.num_workers > 1:
            self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self._executor = None

    def shutdown(self):
        """Shut down the worker pool."""
        if self._executor:
            self._executor.shutdown(wait=False)

    def _extract_weights(self, brain: SnakeBrain) -> List[np.ndarray]:
        """Extract weights from a brain as numpy arrays."""
        return [p.data.numpy().copy() for p in brain.parameters()]

    def _inject_weights(self, brain: SnakeBrain, weights: List[np.ndarray]):
        """Inject weights into a brain."""
        for param, weight in zip(brain.parameters(), weights):
            param.data = torch.from_numpy(weight.copy())

    def _crossover(self, parent1_weights: List[np.ndarray],
                  parent2_weights: List[np.ndarray]) -> List[np.ndarray]:
        """Perform crossover between two parents' weights."""
        child_weights = []
        for w1, w2 in zip(parent1_weights, parent2_weights):
            mask = np.random.rand(*w1.shape) < 0.5
            child_w = np.where(mask, w1, w2).copy()
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
        gen_start = time.time()

        # Extract weights as Python lists (lightweight, fast to pickle)
        work_items = [
            (*brain.get_fast_weights(), board_size, 20)
            for brain in self.population
        ]

        if self._executor is not None:
            # Parallel evaluation
            results = list(self._executor.map(_evaluate_brain_worker, work_items))
        else:
            # Sequential evaluation (--workers 1)
            results = [_fast_evaluate(*item) for item in work_items]

        scores = [r[0] for r in results]
        game_records = [r[1] for r in results]

        gen_elapsed = time.time() - gen_start
        mode = f"{self.num_workers} workers" if self.num_workers > 1 else "sequential"
        print(f"  Evaluated {self.population_size} brains in {gen_elapsed:.2f}s ({mode})")

        # Sort population, scores, and game records together
        sorted_data = sorted(zip(scores, self.population, game_records),
                           key=lambda x: x[0], reverse=True)
        scores, self.population, game_records = zip(*sorted_data)
        scores = list(scores)
        self.population = list(self.population)
        game_records = list(game_records)

        # Visualize best snake (main process only)
        if visualizer:
            print("\nReplicating best snake's performance...")
            game = SnakeGame(board_size=board_size)
            game.reset()
            best_record = simulate_game(self.population[0], game, visualizer=visualizer)
            game_records[0] = best_record

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
            best_game=game_records[0]
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evolve neural networks to play Snake (optimized)")

    # Evolution parameters
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run (default: 100)")
    parser.add_argument("--population", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count). Use 1 for sequential.")

    # Visualization parameters
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--show-every", type=int, default=5, help="Show best game every N generations (default: 5)")
    parser.add_argument("--show-after", type=int, default=15, help="Start showing replays after generation N (default: 15)")
    parser.add_argument("--only-new", action="store_true", help="Only replay when the best game has improved since last replay")

    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save results (default: ./output)")

    # Load and replay modes
    parser.add_argument("--load-brain", type=str, default=None, help="Path to a best_brain.pt file to play games with")
    parser.add_argument("--num-games", type=int, default=1, help="Number of games to play with loaded brain (default: 1)")
    parser.add_argument("--replay", type=str, default=None, help="Path to a best_game.json file to replay")

    args = parser.parse_args()
    board_size = 20

    # --- Replay mode ---
    if args.replay:
        _ensure_pygame()
        print(f"Loading game record from {args.replay}")
        game_record = load_game_record(args.replay)
        print(f"Game score: {game_record.final_score}, steps: {game_record.total_steps}")
        visualizer = SnakeVisualizer(cell_size=30)
        visualizer.replay_game(game_record, speed_ms=50)
        return

    # --- Load brain mode ---
    if args.load_brain:
        _ensure_pygame()
        print(f"Loading brain from {args.load_brain}")
        brain = SnakeBrain(board_size)
        brain.load_state_dict(torch.load(args.load_brain, weights_only=True))
        brain.eval()

        visualizer = SnakeVisualizer(cell_size=30)
        visualizer.setup_display(board_size)

        try:
            for i in range(args.num_games):
                game = SnakeGame(board_size=board_size)
                game.reset()
                print(f"\nGame {i + 1}/{args.num_games}")
                record = simulate_game(brain, game, visualizer=visualizer, speed_ms=100)
                if record is None:
                    break
                print(f"Score: {record.final_score}, Steps: {record.total_steps}")
        finally:
            visualizer.cleanup()
        return

    # --- Evolution mode ---
    show_visualization = not args.no_viz

    # Create genetic algorithm
    ga = GeneticAlgorithm(population_size=args.population, num_workers=args.workers)

    # Create visualizer only if visualization is enabled
    visualizer = None
    if show_visualization:
        _ensure_pygame()
        visualizer = SnakeVisualizer(cell_size=30)

    # Keep track of best game ever and best brain
    best_game_ever = None
    best_score_ever = float('-inf')
    best_brain_ever = None
    last_replayed_score = float('-inf')

    try:
        # Evolution loop
        for gen in range(args.generations):
            # Evolve population (evaluate without visualization)
            stats = ga.evolve_population(board_size, None)

            # Check if this is the best game ever
            if stats.best_score > best_score_ever:
                best_score_ever = stats.best_score
                best_game_ever = stats.best_game
                best_brain_ever = copy.deepcopy(ga.population[0])

            # Print statistics
            print(f"\nGeneration {stats.generation} Statistics:")
            print(f"Best Score: {stats.best_score:.2f}")
            print(f"Average Score: {stats.average_score:.2f}")
            print(f"Worst Score: {stats.worst_score:.2f}")

            # Show replay of the best game if visualization is enabled
            if show_visualization and best_game_ever and gen % args.show_every == 0 and gen > args.show_after:
                if not args.only_new or best_score_ever > last_replayed_score:
                    print("Replaying best game ever...")
                    visualizer.replay_game(best_game_ever, speed_ms=50)
                    last_replayed_score = best_score_ever
                    time.sleep(1)  # Pause between generations

    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
    finally:
        ga.shutdown()
        if visualizer:
            visualizer.cleanup()

    # Save results
    if best_brain_ever and best_game_ever:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        brain_path = os.path.join(run_dir, "best_brain.pt")
        game_path = os.path.join(run_dir, "best_game.json")

        torch.save(best_brain_ever.state_dict(), brain_path)
        save_game_record(best_game_ever, game_path)

        print(f"\nResults saved to {run_dir}/")
        print(f"  Best brain: {brain_path}")
        print(f"  Best game:  {game_path} (score: {best_score_ever:.2f})")


if __name__ == "__main__":
    main()
