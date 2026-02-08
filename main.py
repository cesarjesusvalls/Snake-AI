from visualization import *

def create_sample_game():
    # Create a simple game where the snake moves in a square pattern
    # and eats some food along the way
    actions = []
    food_positions = []
    
    # Start at center, moving right
    initial_position = (10, 10)
    initial_direction = (1, 0)
    
    # Create a pattern that shows growth clearly
    # Move right, then down in a spiral pattern
    actions = (
        [Action.RIGHT] * 4 +    # Move right
        [Action.DOWN] * 4 +     # Move down
        [Action.LEFT] * 4 +     # Move left
        [Action.UP] * 4         # Move up
    ) * 3                       # Repeat pattern three times
    
    # Place food along the path
    food_positions = [
        (13, 10),  # Right path
        (14, 13),  # Down path
        (11, 14),  # Left path
        (10, 11),  # Up path
        (12, 10),  # Second round
        (13, 12),  # More food positions
        (11, 13),
        (11, 11)
    ]
    
    return GameRecord(
        initial_position=initial_position,
        initial_direction=initial_direction,
        actions=actions,
        food_positions=food_positions,
        final_score=8,
        total_steps=len(actions)
    )

def main():
    # Create and replay a sample game
    game_record = create_sample_game()
    visualizer = SnakeVisualizer(cell_size=30)
    visualizer.replay_game(game_record, speed_ms=150)  # Slowed down slightly to see growth better

if __name__ == "__main__":
    main()