from typing import Tuple

import numpy as np

from cogrip.constants import COLOR_NAME_TO_IDX, SHAPE_NAME_TO_IDX
from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board
from cogrip.pentomino.symbolic.types import RelPositions

COLORS_TO_NUMPY = {
    "gripper": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "green": np.array([0, 128, 0]),
    "blue": np.array([0, 0, 255]),
    "cyan": np.array([0, 255, 255]),
    "purple": np.array([128, 0, 128]),
    "brown": np.array([139, 69, 19]),
    "grey": np.array([128, 128, 128]),
    "pink": np.array([255, 192, 203]),
    "olive green": np.array([128, 128, 0]),
    "navy blue": np.array([0, 0, 128]),
}


def pad_with_zeros_to_center(center_pos: Tuple[int, int], array: np.array, max_size: int, channel_first: bool = True):
    #if max_size % 2 == 0:
    #    max_size = max_size + 1 # add one to allow "center" position
    cx, cy = center_pos
    # zeros are black or out-of-world symbols
    if channel_first:
        height, width = array.shape[1:]
    else:
        height, width = array.shape[:2]
    x_left = max_size - cx
    x_right = max_size - (width - cx)
    y_top = max_size - cy
    y_bottom = max_size - (height - cy)
    paddings = [(y_top, y_bottom), (x_left, x_right)]
    if channel_first:
        paddings.insert(0, (0, 0))  # pre-pend no-padding on channel axis
    else:
        paddings.append((0, 0))  # append no-padding on channel axis
    array = np.pad(array, mode="constant", pad_width=paddings)
    return array


def pad_with_zeros_to_square(array, padding: int = 0, channel_first: bool = True):
    # zeros are black or out-of-world symbols
    if padding > 0:
        pad_left = int(padding / 2)
        pad_right = int(padding / 2)
        if padding % 2 != 0:  # add one for odd paddings
            pad_right += 1
        paddings = [(pad_left, pad_right)] * (len(array.shape) - 1)
        if paddings:
            if channel_first:
                paddings.insert(0, (0, 0))  # pre-pend no-padding
            else:
                paddings.append((0, 0))
            array = np.pad(array, mode="constant", pad_width=paddings)
    return array


def compute_pos_mask(position: RelPositions, width: int, height: int):
    pos_mask = np.zeros(shape=(1, height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            symbolic_pos = RelPositions.from_coords(x, y, width, height)
            if symbolic_pos == position:
                pos_mask[0, y, x] = 255
    return pos_mask


def compute_pieces_mask(board: Board, channel_first: bool = True):
    max_size = board.grid_config.width
    piece_mask = np.full((1, max_size, max_size), fill_value=0, dtype=np.uint8)
    for y in range(max_size):
        for x in range(max_size):
            tile = board.object_grid.grid[y][x]
            if tile.objects:
                piece_mask[:, y, x] = 255
    if not channel_first:
        piece_mask = np.moveaxis(piece_mask, 0, -1)
    return piece_mask


def compute_target_mask(board: Board, target_piece: Piece, channel_first: bool = True):
    max_size = board.grid_config.width
    target_mask = np.full((1, max_size, max_size), fill_value=0, dtype=np.uint8)
    for y in range(max_size):
        for x in range(max_size):
            tile = board.object_grid.grid[y][x]
            if tile.objects:
                obj_id = tile.objects[0].id_n
                if target_piece.id_n == obj_id:
                    target_mask[:, y, x] = 255
    if not channel_first:
        target_mask = np.moveaxis(target_mask, 0, -1)
    return target_mask


def surrounding_tiles(y, x, board):
    tile_above, tile_below, tile_left, tile_right = None, None, None, None
    if y > 0:
        tile_above = board.object_grid.grid[y - 1][x]
    if y < board.grid_config.height - 1:
        tile_below = board.object_grid.grid[y + 1][x]
    if x > 0:
        tile_left = board.object_grid.grid[y][x - 1]
    if x < board.grid_config.width - 1:
        tile_right = board.object_grid.grid[y][x + 1]
    return tile_above, tile_below, tile_left, tile_right


def do_draw(oy, ox, y, x, obj_id, board):
    tile_above, tile_below, tile_left, tile_right = surrounding_tiles(y, x, board)
    if oy == 0 and ox == 0:
        return False
    if oy == 0:
        return tile_above and tile_above.objects and tile_above.objects[0].id_n == obj_id
    if ox == 0:
        return tile_left and tile_left.objects and tile_left.objects[0].id_n == obj_id
    return True


def to_rgb_array(board: Board, gripper_coords: Tuple[int, int], channel_first: bool = True, margin: int = 1):
    # when margin=1 returns a 5 times larger array
    # draws white borders around objects, which requires to expand the object to 3 tiles + 1 to the sides
    board_size = board.grid_config.width
    out_size = board_size
    if margin > 0:
        out_size = board_size * 5
    rgb_array = np.full((3, out_size, out_size), fill_value=255, dtype=np.uint8)  # fill with white
    py, px = 0, 0
    for y in range(board_size):
        for x in range(board_size):
            tile = board.object_grid.grid[y][x]
            if tile.objects:
                obj_id = tile.objects[0].id_n
                piece = board.get_piece(obj_id)
                symbol = piece.piece_config
                color_name = symbol.color.value_name
                tile_color = COLORS_TO_NUMPY[color_name]
                if margin > 0:
                    # project the pixels onto larger map
                    for oy in range(5):
                        for ox in range(5):
                            if do_draw(oy, ox, y, x, obj_id, board):
                                rgb_array[:, py + oy, px + ox] = tile_color
                else:
                    rgb_array[:, y, x] = tile_color
            px += 5
        px = 0
        py += 5
    grx, gry = gripper_coords
    if margin > 0:
        gry *= 5
        grx *= 5
        for offset_y in range(1, 5):
            for offset_x in range(1, 5):
                rgb_array[:, gry + offset_y, grx + offset_x] = COLORS_TO_NUMPY["gripper"]
    else:
        rgb_array[:, gry, grx] = COLORS_TO_NUMPY["gripper"]

    if not channel_first:
        rgb_array = np.moveaxis(rgb_array, 0, -1)
    return rgb_array


def to_symbolic_array(board: Board, channel_first: bool = True):
    max_size = board.grid_config.width
    symbolic_array = np.full((3, max_size, max_size), fill_value=1, dtype=np.uint8)  # fill with white
    for y in range(max_size):
        for x in range(max_size):
            tile = board.object_grid.grid[y][x]
            if tile.objects:
                obj_id = tile.objects[0].id_n
                piece = board.get_piece(obj_id)
                symbol = piece.piece_config
                color_name = symbol.color.value_name
                symbolic_array[0, y, x] = COLOR_NAME_TO_IDX[color_name]
                shape_name = symbol.shape.value
                symbolic_array[1, y, x] = SHAPE_NAME_TO_IDX[shape_name]
                symbolic_array[2, y, x] = obj_id
    if not channel_first:
        symbolic_array = np.moveaxis(symbolic_array, 0, -1)
    return symbolic_array


def compute_fov(full_view: np.array, gripper_coords: Tuple[int, int], fov_size: int, channel_first: bool = True,
                margin: int = 1):
    """ Given the full vision obs slice out a fov sized view around the gripper coords """
    x, y = gripper_coords
    gripper_size = 1
    if margin > 0:
        fov_size *= 5  # 35
        x *= 5
        y *= 5
        gripper_size = 5

    context_size = int((fov_size - gripper_size) / 2)  # 15
    topx = x - context_size
    topy = y - context_size

    if channel_first:
        # fill with black or "out-of-world"
        num_channels = full_view.shape[0]
        partial_view = np.full((num_channels, fov_size, fov_size), fill_value=0)
    else:
        # fill with black or "out-of-world"
        num_channels = full_view.shape[-1]
        partial_view = np.full((fov_size, fov_size, num_channels), fill_value=0)
    map_size = full_view.shape[1]  # in both cases channel first (CxHxW) or last (HxWxC) b.c. W==H
    for offy in range(fov_size):
        for offx in range(fov_size):
            vx = topx + offx
            vy = topy + offy
            if (vx >= 0 and vy >= 0) and (vx < map_size and vy < map_size):
                if channel_first:
                    arr = full_view[:, vy, vx]
                    partial_view[:, offy, offx] = arr
                else:
                    arr = full_view[vy, vx]
                    partial_view[offy, offx, :] = arr
    return partial_view
