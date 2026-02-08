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

        # Update direction based on action
        if action != Action.NOTHING:
            new_direction = self.direction_map[action]
            # Prevent 180-degree turns
            if (new_direction[0] != -self.current_direction[0] or 
                new_direction[1] != -self.current_direction[1]):
                self.current_direction = new_direction

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
        if self.steps_without_food >= 50:
            self.game_over = True
            return False

        self.steps += 1
        return True

class SnakeBrain(nn.Module):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size
        
        # Calculate input size:
        # board_size * board_size * 3 (for body, head, food)
        # + 4 (for direction one-hot encoding)
        input_size = board_size * board_size * 3 + 4
        
        # Simple feedforward network
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)  # 5 outputs for L, R, U, D, N
        
        # Initialize with random weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1, 1)
    
    def _direction_to_one_hot(self, direction: Tuple[int, int]) -> torch.Tensor:
        """Convert direction tuple to one-hot encoded tensor."""
        # Map (dx, dy) to index: (0,-1)=0, (1,0)=1, (0,1)=2, (-1,0)=3
        direction_to_idx = {
            (0, -1): 0,  # Up
            (1, 0): 1,   # Right
            (0, 1): 2,   # Down
            (-1, 0): 3   # Left
        }
        idx = direction_to_idx[direction]
        one_hot = torch.zeros(4)
        one_hot[idx] = 1
        return one_hot
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def get_action(self, occupancy_map: np.ndarray, current_direction: Tuple[int, int]) -> Action:
        """Convert network output to snake action."""
        # Create flattened input tensor
        # First, separate occupancy map into channels and flatten
        body = (occupancy_map == 1).astype(np.float32).flatten()
        head = (occupancy_map == 2).astype(np.float32).flatten()
        food = (occupancy_map == 3).astype(np.float32).flatten()
        
        # Convert to tensors and concatenate
        state_tensor = torch.cat([
            torch.from_numpy(body),
            torch.from_numpy(head),
            torch.from_numpy(food),
            self._direction_to_one_hot(current_direction)
        ])
        
        # Add batch dimension
        x = state_tensor.unsqueeze(0)
        
        # Get network prediction
        with torch.no_grad():
            action_probs = self(x)
        
        # Convert to numpy and get action index
        action_idx = action_probs.numpy().argmax()
        
        # Map index to action
        actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.NOTHING]
        return actions[action_idx]

class SnakeVisualizer:
    def __init__(self, cell_size: int = 20):
        self.cell_size = cell_size
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.colors = {
            'background': (0, 0, 0),
            'snake': (0, 255, 0),
            'food': (255, 0, 0),
            'head': (0, 200, 0),
            'text': (255, 255, 255)
        }

    def setup_display(self, board_size: int):
        """Initialize the display with given board size."""
        self.board_size = board_size
        self.window_size = board_size * self.cell_size
        # Make window wider to accommodate debug info
        self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
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
        
        # Draw grid
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (0, i * self.cell_size), 
                           (self.window_size, i * self.cell_size))

    def draw_info(self, text: str):
        """Draw debug information on the right side of the screen."""
        x = self.window_size + 10
        y = 10
        for line in text.split('\n'):
            text_surface = self.font.render(line, True, self.colors['text'])
            self.screen.blit(text_surface, (x, y))
            y += 20
        pygame.display.flip()

def simulate_game(brain: SnakeBrain, game: SnakeGame, visualizer: Optional[SnakeVisualizer] = None, 
                 speed_ms: int = 100) -> GameRecord:
    """Simulate a full game using the given brain."""
    actions = []
    food_positions = [game.food_position]
    
    clock = pygame.time.Clock() if visualizer else None
    
    while not game.game_over:
        # Get current state
        occupancy = game.get_occupancy_map()
        
        # Get action from brain
        action = brain.get_action(occupancy, game.current_direction)
        actions.append(action)
        
        # Show current game state if visualizer is provided
        if visualizer:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                    
            visualizer.draw_state(game.get_state())
            visualizer.draw_info(
                f"Score: {game.score}  Steps: {game.steps}\n"
                f"Direction: {game.current_direction}\n"
                f"Action: {action.value}\n"
                f"Steps without food: {game.steps_without_food}"
            )
            clock.tick(1000 / speed_ms)
        
        # Apply action and check if game is over
        if not game.step(action):
            if visualizer:
                # Show final state and wait a bit
                visualizer.draw_state(game.get_state())
                visualizer.draw_info(
                    f"GAME OVER!\n"
                    f"Final Score: {game.score}  Steps: {game.steps}\n"
                    f"Hit a wall: {game.hit_wall}\n"
                    f"Self collision: {game.self_collision}\n"
                    f"Stuck: {game.steps_without_food >= 50}"
                )
                pygame.display.flip()
                time.sleep(2)  # Show end state for 2 seconds
            break
            
        # Record food position if it changed (was eaten)
        if len(actions) > 1 and game.food_position != food_positions[-1]:
            food_positions.append(game.food_position)
    
    return GameRecord(
        initial_position=game.snake_positions[0],
        initial_direction=game.current_direction,
        actions=actions,
        food_positions=food_positions,
        final_score=game.score,
        total_steps=game.steps
    )

def main():
    # Create a game instance
    board_size = 20
    game = SnakeGame(board_size=board_size)
    
    # Create a random brain
    brain = SnakeBrain(board_size)
    
    # Create visualizer
    visualizer = SnakeVisualizer(cell_size=30)
    visualizer.setup_display(board_size)
    
    # Simulate multiple games
    num_games = 5
    best_score = 0
    best_record = None
    
    try:
        for i in range(num_games):
            game.reset()
            print(f"\nStarting Game {i+1}")
            game_record = simulate_game(brain, game, visualizer=visualizer, speed_ms=100)
            
            if game_record is None:  # Game was quit by user
                break
                
            print(f"Game {i+1} - Score: {game_record.final_score}, Steps: {game_record.total_steps}")
            
            if game_record.final_score > best_score:
                best_score = game_record.final_score
                best_record = game_record
            
            time.sleep(1)  # Pause between games
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()