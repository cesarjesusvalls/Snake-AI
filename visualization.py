import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import random

class Action(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    UP = 'U'
    DOWN = 'D'
    NOTHING = 'N'

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
        
        # Direction mappings
        self.direction_map = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
            Action.NOTHING: None  # Will continue in current direction
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
        self.growth_remaining = 0  # Track remaining growth after eating food

    def _spawn_food(self, position: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Spawn food at given position or random position if None."""
        if position is not None:
            return position
            
        available_positions = [
            (x, y) for x in range(self.board_size) for y in range(self.board_size)
            if (x, y) not in self.snake_positions
        ]
        return random.choice(available_positions) if available_positions else (0, 0)

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
        for x, y in self.snake_positions:
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

        # Update direction based on action
        if action != Action.NOTHING:
            new_direction = self.direction_map[action]
            # Prevent 180-degree turns
            if (new_direction[0] != -self.current_direction[0] or 
                new_direction[1] != -self.current_direction[1]):
                self.current_direction = new_direction

        # Calculate new head position
        head_x, head_y = self.snake_positions[-1]
        new_x = (head_x + self.current_direction[0]) % self.board_size
        new_y = (head_y + self.current_direction[1]) % self.board_size
        new_head = (new_x, new_y)

        # Check collision with self
        if new_head in self.snake_positions:
            self.game_over = True
            return False

        # Move snake
        self.snake_positions.append(new_head)
        
        # Check if food is eaten
        if new_head == self.food_position:
            self.score += 1
            self.steps_without_food = 0
            self.growth_remaining += 1  # Add growth when food is eaten
            self.food_position = self._spawn_food()
        
        # Handle snake growth
        if self.growth_remaining > 0:
            self.growth_remaining -= 1  # Decrease remaining growth
        else:
            self.snake_positions.pop(0)  # Only remove tail if not growing
            
        # Check if snake is stuck
        if self.steps_without_food >= 50:
            self.game_over = True
            return False

        self.steps += 1
        return True

class SnakeVisualizer:
    def __init__(self, cell_size: int = 20):
        self.cell_size = cell_size
        pygame.init()
        self.colors = {
            'background': (0, 0, 0),
            'snake': (0, 255, 0),
            'food': (255, 0, 0),
            'head': (0, 200, 0)
        }

    def setup_display(self, board_size: int):
        """Initialize the display with given board size."""
        self.window_size = board_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Snake Game')

    def draw_state(self, state: GameState):
        """Draw the current game state."""
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
        
        pygame.display.flip()

    def replay_game(self, game_record: GameRecord, speed_ms: int = 100):
        """Replay a recorded game."""
        game = SnakeGame(board_size=20, cell_size=self.cell_size)
        game.reset(game_record.initial_position, game_record.initial_direction)
        
        food_position_index = 0
        game.food_position = game_record.food_positions[food_position_index]

        self.setup_display(game.board_size)
        clock = pygame.time.Clock()

        for action in game_record.actions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if game.step(action):
                # Check if food was eaten by comparing snake length
                if (len(game.snake_positions) > food_position_index + 1):
                    food_position_index += 1
                    if food_position_index < len(game_record.food_positions):
                        game.food_position = game_record.food_positions[food_position_index]

                self.draw_state(game.get_state())
                clock.tick(1000 / speed_ms)
            else:
                break

        pygame.quit()