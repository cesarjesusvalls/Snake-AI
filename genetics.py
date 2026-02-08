"""smart_genetics.py — SnakeAI neuroevolution with spatial awareness

Enhanced brain inputs over genetics.py:
  - Tail direction (straight/left/right) — follow your tail to avoid self-trapping
  - Flood fill reachable space (straight/left/right) — BFS from each next position,
    normalized by snake length. Detects dead-end pockets before entering them.

Network architecture: 12 inputs → 16 hidden (ReLU) → 3 outputs (LEFT/RIGHT/NOTHING)
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
        if self.steps_without_food >= 150:
            self.game_over = True
            return False

        self.steps += 1
        return True


# ---------------------------------------------------------------------------
# SnakeBrain — 12 inputs → 16 hidden (ReLU) → 3 outputs
# ---------------------------------------------------------------------------

class SnakeBrain(nn.Module):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size

        # Input features (12 inputs):
        # 1-3:  Food direction (straight, left, right)
        # 4-6:  Death checks (straight, left, right)
        # 7-9:  Tail direction (straight, left, right)
        # 10-12: Reachable space ratio (straight, left, right)
        input_size = 12
        hidden_size = 16

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)  # 3 outputs for LEFT, RIGHT, NOTHING

        # Initialize with random weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1, 1)

    def get_fast_weights(self):
        """Extract weights as Python lists for fast inference.

        Returns (wh, bh, wo, bo):
          wh: hidden weights (16x12 nested list)
          bh: hidden biases (length-16 list)
          wo: output weights (3x16 nested list)
          bo: output biases (length-3 list)
        """
        return (
            self.fc1.weight.data.tolist(),
            self.fc1.bias.data.tolist(),
            self.fc2.weight.data.tolist(),
            self.fc2.bias.data.tolist(),
        )

    def _flood_fill_count(self, start_x: int, start_y: int,
                          blocked: set, max_count: int) -> int:
        """BFS flood fill counting reachable cells. Stops at max_count."""
        bs = self.board_size
        if (start_x < 0 or start_x >= bs or start_y < 0 or start_y >= bs
                or (start_x, start_y) in blocked):
            return 0
        visited = {(start_x, start_y)}
        queue = deque([(start_x, start_y)])
        count = 1
        while queue:
            if count >= max_count:
                return count
            x, y = queue.popleft()
            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if (0 <= nx < bs and 0 <= ny < bs
                        and (nx, ny) not in blocked and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    count += 1
                    if count >= max_count:
                        return count
        return count

    def _get_input_features(self, occupancy_map: np.ndarray,
                            current_direction: Tuple[int, int],
                            snake_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """Get 12 input features for the neural network."""
        bs = self.board_size

        # Find head and food positions
        hy, hx = np.where(occupancy_map == 2)
        head_pos = (int(hx[0]), int(hy[0]))
        fy, fx = np.where(occupancy_map == 3)
        food_pos = (int(fx[0]), int(fy[0]))
        tail_pos = snake_positions[0]

        dx, dy = current_direction
        lx, ly = -dy, dx    # left direction
        rx, ry = dy, -dx    # right direction

        # --- Food direction (dot/cross product) ---
        fdx = food_pos[0] - head_pos[0]
        fdy = food_pos[1] - head_pos[1]
        fwd = fdx * dx + fdy * dy
        lft = fdx * lx + fdy * ly
        a_lft, a_fwd = abs(lft), abs(fwd)
        food_straight = float(fwd > 0 and a_lft < a_fwd)
        food_left = float(lft > 0 and a_fwd < a_lft)
        food_right = float(lft < 0 and a_fwd < a_lft)

        # --- Death checks ---
        def is_death(px, py):
            return (px < 0 or px >= bs or py < 0 or py >= bs
                    or occupancy_map[py, px] == 1)
        death_straight = float(is_death(head_pos[0] + dx, head_pos[1] + dy))
        death_left = float(is_death(head_pos[0] + lx, head_pos[1] + ly))
        death_right = float(is_death(head_pos[0] + rx, head_pos[1] + ry))

        # --- Tail direction (dot/cross product, same as food) ---
        tdx = tail_pos[0] - head_pos[0]
        tdy = tail_pos[1] - head_pos[1]
        t_fwd = tdx * dx + tdy * dy
        t_lft = tdx * lx + tdy * ly
        a_t_lft, a_t_fwd = abs(t_lft), abs(t_fwd)
        tail_straight = float(t_fwd > 0 and a_t_lft < a_t_fwd)
        tail_left = float(t_lft > 0 and a_t_fwd < a_t_lft)
        tail_right = float(t_lft < 0 and a_t_fwd < a_t_lft)

        # --- Flood fill (reachable space ratio) ---
        snake_set = set(snake_positions)
        snake_len = len(snake_positions)
        space_s = min(self._flood_fill_count(
            head_pos[0] + dx, head_pos[1] + dy, snake_set, snake_len) / snake_len, 1.0)
        space_l = min(self._flood_fill_count(
            head_pos[0] + lx, head_pos[1] + ly, snake_set, snake_len) / snake_len, 1.0)
        space_r = min(self._flood_fill_count(
            head_pos[0] + rx, head_pos[1] + ry, snake_set, snake_len) / snake_len, 1.0)

        return torch.tensor([
            food_straight, food_left, food_right,
            death_straight, death_left, death_right,
            tail_straight, tail_left, tail_right,
            space_s, space_l, space_r,
        ], dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def get_action(self, occupancy_map: np.ndarray,
                   current_direction: Tuple[int, int],
                   snake_positions: List[Tuple[int, int]]) -> Action:
        """Convert network output to snake action."""
        input_features = self._get_input_features(
            occupancy_map, current_direction, snake_positions)

        x = input_features.unsqueeze(0)
        with torch.no_grad():
            action_probs = self(x)

        action_idx = action_probs.numpy().argmax()
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
        action = brain.get_action(occupancy, game.current_direction, game.snake_positions)
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
# Flood fill — module-level for worker pickling
# ---------------------------------------------------------------------------

def _flood_fill(start_x, start_y, bs, snake_set, max_count):
    """BFS flood fill counting reachable cells from (start_x, start_y).

    Treats walls and snake_set cells as impassable. Stops early once
    reachable count reaches max_count.

    Returns integer count of reachable cells (0 if start is blocked/OOB).
    """
    if (start_x < 0 or start_x >= bs or start_y < 0 or start_y >= bs
            or (start_x, start_y) in snake_set):
        return 0
    visited = {(start_x, start_y)}
    queue = deque([(start_x, start_y)])
    count = 1
    while queue:
        if count >= max_count:
            return count
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nx < bs and 0 <= ny < bs
                    and (nx, ny) not in snake_set and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
                count += 1
                if count >= max_count:
                    return count
    return count


# ---------------------------------------------------------------------------
# Fast evaluation — inlined game loop + numpy inference
# ---------------------------------------------------------------------------

def _fast_evaluate(wh, bh, wo, bo, board_size, num_trials):
    """Evaluate a brain over multiple game trials.

    Args:
        wh: hidden layer weights as nested list (16x12)
        bh: hidden layer biases as list (16)
        wo: output layer weights as nested list (3x16)
        bo: output layer biases as list (3)
        board_size: board dimension (e.g. 20)
        num_trials: number of games to average over

    Returns:
        (avg_score, best_game_record)
    """
    # Convert weights to numpy once (numpy matmul faster than Python for 12→16→3)
    wh_np = np.array(wh, dtype=np.float32)   # (16, 12)
    bh_np = np.array(bh, dtype=np.float32)   # (16,)
    wo_np = np.array(wo, dtype=np.float32)   # (3, 16)
    bo_np = np.array(bo, dtype=np.float32)   # (3,)

    bs = board_size
    bs_minus_1 = bs - 1
    start_pos = (bs // 2, bs // 2)
    randint = random.randint

    total_score = 0.0
    best_fitness = -1.0
    best_food_score = 0
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
        total_steps = 0
        steps_without_food = 0
        growth = 0
        actions = []
        food_positions = [(food_x, food_y)]
        steps_at_last_food = 0

        while True:
            hx, hy = snake[-1]
            snake_len = len(snake)

            # --- Food direction (dot/cross product) ---
            fdx = food_x - hx
            fdy = food_y - hy
            fwd = fdx * dx + fdy * dy
            lx, ly = -dy, dx
            lft = fdx * lx + fdy * ly
            a_lft = abs(lft)
            a_fwd = abs(fwd)
            fs = 1.0 if fwd > 0 and a_lft < a_fwd else 0.0
            fl = 1.0 if lft > 0 and a_fwd < a_lft else 0.0
            fr = 1.0 if lft < 0 and a_fwd < a_lft else 0.0

            # --- Death checks (set-based O(1)) ---
            nx, ny = hx + dx, hy + dy
            ds = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0
            nx, ny = hx + lx, hy + ly
            dl = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0
            rx, ry = dy, -dx
            nx, ny = hx + rx, hy + ry
            dr = 1.0 if (nx < 0 or nx > bs_minus_1 or ny < 0 or ny > bs_minus_1
                         or (nx, ny) in snake_set) else 0.0

            # --- Tail direction (dot/cross product) ---
            tx, ty = snake[0]
            tdx = tx - hx
            tdy = ty - hy
            t_fwd = tdx * dx + tdy * dy
            t_lft = tdx * lx + tdy * ly
            a_t_lft = abs(t_lft)
            a_t_fwd = abs(t_fwd)
            ts = 1.0 if t_fwd > 0 and a_t_lft < a_t_fwd else 0.0
            tl = 1.0 if t_lft > 0 and a_t_fwd < a_t_lft else 0.0
            tr = 1.0 if t_lft < 0 and a_t_fwd < a_t_lft else 0.0

            # --- Flood fill (reachable space / snake_length, capped at 1.0) ---
            ss = min(_flood_fill(hx + dx, hy + dy, bs, snake_set, snake_len) / snake_len, 1.0)
            sl = min(_flood_fill(hx + lx, hy + ly, bs, snake_set, snake_len) / snake_len, 1.0)
            sr = min(_flood_fill(hx + rx, hy + ry, bs, snake_set, snake_len) / snake_len, 1.0)

            # --- Forward pass (numpy matmul: 12 → 16 ReLU → 3) ---
            inp = np.array([fs, fl, fr, ds, dl, dr, ts, tl, tr, ss, sl, sr],
                           dtype=np.float32)
            hidden = wh_np @ inp + bh_np
            np.maximum(hidden, 0, out=hidden)  # ReLU in-place
            output = wo_np @ hidden + bo_np
            best_i = int(output.argmax())

            actions.append(best_i)

            # --- Apply direction change ---
            if best_i == 0:    # LEFT
                dx, dy = -dy, dx
            elif best_i == 1:  # RIGHT
                dx, dy = dy, -dx

            # --- Move ---
            new_hx = hx + dx
            new_hy = hy + dy

            if new_hx < 0 or new_hx > bs_minus_1 or new_hy < 0 or new_hy > bs_minus_1:
                break
            new_head = (new_hx, new_hy)
            if new_head in snake_set:
                break

            snake.append(new_head)
            snake_set.add(new_head)
            total_steps += 1

            if new_hx == food_x and new_hy == food_y:
                score += 1
                steps_without_food = 0
                growth += 1
                steps_at_last_food = total_steps
                while True:
                    fx, fy = randint(0, bs_minus_1), randint(0, bs_minus_1)
                    if (fx, fy) not in snake_set:
                        break
                food_x, food_y = fx, fy
                food_positions.append((food_x, food_y))

            if growth > 0:
                growth -= 1
            else:
                removed = snake.popleft()
                snake_set.discard(removed)
                steps_without_food += 1

            if steps_without_food >= 150:
                break

        # Fitness = score + efficiency bonus in [0, 1)
        # Bonus rewards fewer average steps per food (normalized by starvation limit)
        if score > 0:
            avg_steps_per_food = steps_at_last_food / score
            efficiency_bonus = 10.0 - 10.0*avg_steps_per_food / 150.0
            if efficiency_bonus < 0.0:
                efficiency_bonus = 0.0
        else:
            efficiency_bonus = 0.0
        fitness = score + efficiency_bonus

        total_score += fitness
        if fitness > best_fitness:
            best_fitness = fitness
            best_food_score = score
            best_actions = actions
            best_food_positions = food_positions

    # Build GameRecord for the best game
    action_enums = [Action.LEFT, Action.RIGHT, Action.NOTHING]
    best_record = GameRecord(
        initial_position=start_pos,
        initial_direction=(1, 0),
        actions=[action_enums[i] for i in best_actions],
        food_positions=[tuple(p) for p in best_food_positions],
        final_score=best_food_score,
        total_steps=len(best_actions),
    )

    return total_score / num_trials, best_record


def _evaluate_brain_worker(args):
    """Worker function for parallel brain evaluation."""
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

        if self.num_workers > 1:
            self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self._executor = None

    def shutdown(self):
        """Shut down the worker pool."""
        if self._executor:
            self._executor.shutdown(wait=False)

    def seed_population(self, seed_brain: SnakeBrain, board_size: int):
        """Create initial population from a seed brain: 1 exact copy + mutated clones."""
        seed_weights = self._extract_weights(seed_brain)
        self.population = []
        for i in range(self.population_size):
            brain = SnakeBrain(board_size)
            weights = [w.copy() for w in seed_weights]
            if i > 0:  # keep first one as exact copy
                self._mutate(weights)
            self._inject_weights(brain, weights)
            self.population.append(brain)

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
        if self.generation == 0 and not hasattr(self, 'population'):
            self.population = [SnakeBrain(board_size) for _ in range(self.population_size)]

        # Evaluate all brains
        print(f"\nEvaluating Generation {self.generation}")
        gen_start = time.time()

        # Extract weights as Python lists (4 arrays: wh, bh, wo, bo)
        work_items = [
            (*brain.get_fast_weights(), board_size, 20)
            for brain in self.population
        ]

        if self._executor is not None:
            results = list(self._executor.map(_evaluate_brain_worker, work_items))
        else:
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
            parent1_idx = int(np.random.exponential(scale=self.population_size/25))
            parent2_idx = int(np.random.exponential(scale=self.population_size/25))
            parent1_idx = min(parent1_idx, len(self.population) - 1)
            parent2_idx = min(parent2_idx, len(self.population) - 1)

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            child = SnakeBrain(board_size)
            child_weights = self._crossover(
                self._extract_weights(parent1),
                self._extract_weights(parent2)
            )
            self._mutate(child_weights)
            self._inject_weights(child, child_weights)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

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
    parser = argparse.ArgumentParser(
        description="Evolve neural networks to play Snake (enhanced spatial awareness)")

    # Evolution parameters
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of generations to run (default: 100)")
    parser.add_argument("--population", type=int, default=100,
                        help="Population size (default: 100)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count). Use 1 for sequential.")

    # Visualization parameters
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--show-every", type=int, default=5,
                        help="Show best game every N generations (default: 5)")
    parser.add_argument("--show-after", type=int, default=15,
                        help="Start showing replays after generation N (default: 15)")
    parser.add_argument("--only-new", action="store_true",
                        help="Only replay when the best game has improved since last replay")

    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save results (default: ./output)")

    # Seed brain for evolution
    parser.add_argument("--seed-brain", type=str, default=None,
                        help="Path to a best_brain.pt to seed the initial population (clones + mutations)")

    # Load and replay modes
    parser.add_argument("--load-brain", type=str, default=None,
                        help="Path to a best_brain.pt file to play games with")
    parser.add_argument("--num-games", type=int, default=1,
                        help="Number of games to play with loaded brain (default: 1)")
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to a best_game.json file to replay")

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

    ga = GeneticAlgorithm(population_size=args.population, num_workers=args.workers)

    if args.seed_brain:
        print(f"Seeding population from {args.seed_brain}")
        seed_brain = SnakeBrain(board_size)
        seed_brain.load_state_dict(torch.load(args.seed_brain, weights_only=True))
        ga.seed_population(seed_brain, board_size)

    visualizer = None
    if show_visualization:
        _ensure_pygame()
        visualizer = SnakeVisualizer(cell_size=30)

    best_game_ever = None
    best_score_ever = float('-inf')
    best_brain_ever = None
    last_replayed_score = float('-inf')

    try:
        for gen in range(args.generations):
            stats = ga.evolve_population(board_size, None)

            if stats.best_score > best_score_ever:
                best_score_ever = stats.best_score
                best_game_ever = stats.best_game
                best_brain_ever = copy.deepcopy(ga.population[0])

            print(f"\nGeneration {stats.generation} Statistics:")
            print(f"Best Score: {stats.best_score:.2f}")
            print(f"Average Score: {stats.average_score:.2f}")
            print(f"Worst Score: {stats.worst_score:.2f}")

            if show_visualization and best_game_ever and gen % args.show_every == 0 and gen > args.show_after:
                if not args.only_new or best_score_ever > last_replayed_score:
                    print("Replaying best game ever...")
                    visualizer.replay_game(best_game_ever, speed_ms=50)
                    last_replayed_score = best_score_ever
                    time.sleep(1)

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
