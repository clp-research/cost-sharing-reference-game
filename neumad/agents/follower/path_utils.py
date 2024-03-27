import random
from typing import Tuple

import networkx

from neumad.envs import FollowerAction


def to_move_actions(from_pos, to_pos) -> Tuple[FollowerAction, FollowerAction]:
    g_x, g_y = from_pos
    t_x, t_y = to_pos
    horizontal = None
    if g_x < t_x:
        horizontal = FollowerAction.right
    if g_x > t_x:
        horizontal = FollowerAction.left
    vertical = None
    if g_y < t_y:
        vertical = FollowerAction.down
    if g_y > t_y:
        vertical = FollowerAction.up
    return horizontal, vertical


def to_action(from_pos, to_pos):
    action_h, action_v = to_move_actions(from_pos, to_pos)
    if action_h is not None and action_v is not None:  # this should actually never happen?!
        preference = random.choice([0, 1])
        if preference == 0:  # prefer horizontal movements
            action = action_h
        else:  # prefer vertical movements
            action = action_v
    elif action_h is not None:
        action = action_h
    elif action_v is not None:
        action = action_v
    else:  # both are None (already on target position)
        action = FollowerAction.wait
    return action


def translate_path_to_actions(path):
    if len(path) < 2:
        return [FollowerAction.wait]
    current = path.pop(0)
    next_node = path.pop(0)
    actions = []
    while next_node is not None:
        action = to_action(current, next_node)
        actions.append(action)
        current = next_node
        if path:
            next_node = path.pop(0)
        else:
            next_node = None
    return actions


def compute_shortest_path_in_view(grid_size, target_coord):
    # make sure that target coord is (x,y)!
    grid = networkx.grid_2d_graph(grid_size, grid_size)
    center_pos = int(grid_size / 2), int(grid_size / 2)
    path = networkx.shortest_path(grid, center_pos, target_coord)
    return path


def compute_shortest_path(current_coord: Tuple[int, int], target_coord: Tuple[int, int], grid_size: int):
    # make sure that target coord is (x,y)!
    grid = networkx.grid_2d_graph(grid_size, grid_size)
    path = networkx.shortest_path(grid, current_coord, target_coord)
    return path
