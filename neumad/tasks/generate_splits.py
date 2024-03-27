from cogrip.constants import COLORS_6, ORIENTATIONS, POSITIONS, SHAPES_7
from cogrip.generation import *
import json


def store_splits_to_json(task_splits, file_name="splits.json"):
    storable_splits = dict()
    for split_name, symbolic_pieces in task_splits.items():
        storable_splits[split_name] = [sp.to_json() for sp in symbolic_pieces]

    with open(file_name, "w") as f:
        json.dump(storable_splits, f)
    return file_name


"""
    We want to define training, validation and test splits for the tasks. 
    We do so by calculating all possible target pieces and their positions.
    From this set we leave away a holdout. The remaining "symbolic pieces"
    -- a combination of shape, color and position -- define the possible 
    target pieces (and their positions) for a task.
    
    We split these possible target pieces so that each subset still contains
    all colors, shapes and positions, but different combinations of them. For 
    example, the training set might contain a "red F" but this is never seen
    at the center. Then it will be seen during validation or testing.
"""


def main():
    colors = COLORS_6
    shapes = SHAPES_7
    positions = POSITIONS  # allow all positions

    # compute holdout for color and store holdout in holdout.json
    configs_by_color_split = create_color_holdout(shapes, colors, positions, orientations=ORIENTATIONS)
    check_color_split_counts(configs_by_color_split, len(positions))
    check_piece_colors_per_split(configs_by_color_split)

    piece_symbols_holdout = list(configs_by_color_split["test"])

    piece_symbols_train = list(configs_by_color_split["train"])
    random.shuffle(piece_symbols_train)

    # total: 270=(6*6*9)-54 (holdout)
    # total: 315=(6*7*9)-63 (holdout)
    # total: 405=(6*9*9)-81 (holdout)
    total = len(piece_symbols_train)
    print("Total", total)
    piece_symbols_test = piece_symbols_train[280:]
    piece_symbols_validate = piece_symbols_train[250:280]
    piece_symbols_train = piece_symbols_train[:250]
    print("Test", len(piece_symbols_test), len(piece_symbols_test) / total)
    print("Val", len(piece_symbols_validate), len(piece_symbols_validate) / total)
    print("Train", len(piece_symbols_train), len(piece_symbols_train) / total)
    print("Holdout", len(piece_symbols_holdout))

    task_splits = dict(train=piece_symbols_train, val=piece_symbols_validate, test=piece_symbols_test,
                       holdout=piece_symbols_holdout)

    check_piece_colors_per_split(task_splits)
    check_piece_pos_per_split(task_splits)

    store_splits_to_json(task_splits)


if __name__ == '__main__':
    main()
