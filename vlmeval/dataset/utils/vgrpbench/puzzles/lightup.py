import argparse
import random
from typing import Any, Dict, List, Tuple
import os

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint, ConstraintRowNoRepeat, ConstraintColNoRepeat, ConstraintSubGridNoRepeat

class ConstraintLightUpBulb(Constraint):
    """Ensures that light bulbs don't illuminate each other.
<<<<<<< HEAD

=======
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    This constraint checks that no two light bulbs ('s') can see each other in any straight line
    (horizontally or vertically) without a wall between them. If two bulbs can see each other,
    the constraint fails.
    """
<<<<<<< HEAD

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_bulb"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

=======
    
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_bulb"
    
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        
>>>>>>> f7bcc2c8 (add vgrpbench)
        for row in range(size):
            for col in range(size):
                if board[row][col] == 's':  # Check light sources
                    # Check each direction
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        while 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 'w':  # Wall
                                break
                            if board[nx][ny] == 's':  # Another light
                                return False
                            # Skip undefined (0) and empty ('e') cells
                            nx += dx
                            ny += dy
        return True

class ConstraintLightUpWall(Constraint):
    """Ensures that numbered walls have the correct number of adjacent light bulbs.
<<<<<<< HEAD

=======
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    This constraint verifies that each numbered wall has exactly the specified number of light
    bulbs placed in orthogonally adjacent cells. The constraint fails if:
    1. A numbered wall has more adjacent light bulbs than its number
    2. A numbered wall cannot possibly reach its required number with the remaining undefined cells
    """
<<<<<<< HEAD

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_wall"

=======
    
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_wall"
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]

        wall_numbers = game_state["wall_numbers"]

        if not wall_numbers:
            return True
<<<<<<< HEAD

        size = len(board)

        for row in range(size):
            for col in range(size):

=======
        
        size = len(board)
        
        for row in range(size):
            for col in range(size):
                
>>>>>>> f7bcc2c8 (add vgrpbench)
                if board[row][col] == 'w' and wall_numbers[row][col] != -1:
                    light_count = 0
                    undefined_count = 0

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 's':
                                light_count += 1
                            elif board[nx][ny] == 0:  # Count undefined cells
                                undefined_count += 1
<<<<<<< HEAD

                    # Fail if:
                    # 1. We have too many definite lights, or
                    # 2. We don't have enough potential lights (current + undefined) to reach the required number
                    if (light_count > wall_numbers[row][col] or
=======
                    
                    # Fail if:
                    # 1. We have too many definite lights, or
                    # 2. We don't have enough potential lights (current + undefined) to reach the required number
                    if (light_count > wall_numbers[row][col] or 
>>>>>>> f7bcc2c8 (add vgrpbench)
                        light_count + undefined_count < wall_numbers[row][col]):
                        return False
        return True

class ConstraintLightUpIllumination(Constraint):
    """Ensures that all non-wall cells are illuminated by at least one light bulb.
<<<<<<< HEAD

=======
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    This constraint verifies that every empty cell ('e') is illuminated by at least one light bulb
    or could potentially be illuminated by an undefined cell. For each empty cell, we check in all
    four directions (up, down, left, right) until hitting a wall. If none of these directions
    contain either a light bulb ('s') or an undefined cell (0), then the cell cannot be illuminated
    in any valid solution.
    """
<<<<<<< HEAD

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_illumination"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

=======
    
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_illumination"
    
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        
>>>>>>> f7bcc2c8 (add vgrpbench)
        # For each empty cell ('e'), check if it can be illuminated
        for row in range(size):
            for col in range(size):
                if board[row][col] == 'e':
                    can_be_illuminated = False
<<<<<<< HEAD

=======
                    
>>>>>>> f7bcc2c8 (add vgrpbench)
                    # Check all four directions until hitting a wall
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        while 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 'w':  # Hit a wall, stop checking this direction
                                break
                            if board[nx][ny] == 's' or board[nx][ny] == 0:  # Found light or potential light
                                can_be_illuminated = True
                                break
                            nx += dx
                            ny += dy
<<<<<<< HEAD

                        if can_be_illuminated:  # If we found a light source, no need to check other directions
                            break

                    if not can_be_illuminated:  # If no direction had a light or potential light
                        return False

=======
                        
                        if can_be_illuminated:  # If we found a light source, no need to check other directions
                            break
                    
                    if not can_be_illuminated:  # If no direction had a light or potential light
                        return False
        
>>>>>>> f7bcc2c8 (add vgrpbench)
        return True

class LightUpPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3:
            raise ValueError("Size must be at least 3")
<<<<<<< HEAD

        self.game_name = "lightup"
        self.size = size

        self.constraints = [
            ConstraintLightUpBulb(),
            ConstraintLightUpWall(),
            ConstraintLightUpIllumination()
        ]

=======
            
        self.game_name = "lightup"
        self.size = size
        
        self.constraints = [
            ConstraintLightUpBulb(),   
            ConstraintLightUpWall(),    
            ConstraintLightUpIllumination()    
        ]
        
>>>>>>> f7bcc2c8 (add vgrpbench)
        self.all_possible_values = ['s', 'e']  # 's' for source/light, 'e' for empty

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        board = game_state["board"]
        if board[row][col] in [-1, 1, 2, 3, 4]:  # Wall or numbered wall
            return []
<<<<<<< HEAD

        possible_values = []
        original_value = board[row][col]

=======
            
        possible_values = []
        original_value = board[row][col]
        
>>>>>>> f7bcc2c8 (add vgrpbench)
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
