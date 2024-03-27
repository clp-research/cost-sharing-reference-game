import collections
import random

from stable_baselines3.common.utils import set_random_seed

from cogrip.core.grid import GridConfig
from cogrip.pentomino.state import Board
from cogrip.pentomino.symbolic.sampling import UtteranceTypeOrientedDistractorSetSampler
from cogrip.pentomino.symbolic.types import SymbolicPiece, PropertyNames, SymbolicPieceGroup
from tqdm import tqdm

from cogrip.tasks import Task, store_tasks_to_json
import json


def load_splits_from_json(file_name="splits.json"):
    with open(file_name) as f:
        data = json.load(f)
    splits = dict()
    for split_name, symbolic_pieces in data.items():
        splits[split_name] = [SymbolicPiece.from_json(sp) for sp in symbolic_pieces]
    return splits


def print_check(tasks, split_name):
    piece_symbols = set()
    num_counts = collections.defaultdict(int)
    for task in tasks[split_name]:
        for p in task.piece_symbols:
            piece_symbols.add(p)
        num_counts[len(task.piece_symbols)] += 1
    print("Tasks:", len(tasks[split_name]))
    print("PCS:", len(piece_symbols))
    print("Possible: 324 (6 * 6 * 9)")
    for k, v in num_counts.items():
        print(k, v)


def check_problems(tasks, split_name):
    print("Total", len(tasks[split_name]))
    count = 0
    for t in tasks[split_name]:
        if t.target_piece_symbol not in t.piece_symbols:
            count += 1
    print("Problems:", count)
    print("-" * 20)


def create_training_task(target_symbol: SymbolicPiece, distractor_symbols: SymbolicPieceGroup,
                         map_size: int, retries: int = 100):
    # create a board and place pieces on it
    grid_config = GridConfig(map_size, map_size, move_step=1, prevent_overlap=True)
    board_symbols = [target_symbol] + distractor_symbols.pieces
    all_success = False
    counter = 0
    while not all_success:  # try to place all pieces
        board = Board(grid_config)
        for symbol in board_symbols:
            all_success, piece = board.add_piece_from_symbol(symbol, max_attempts=100, verbose=False)
            if not all_success:
                counter += 1
                if counter > retries:
                    raise Exception(f"Too many tries for {(piece.piece_config, piece.x, piece.y)}: "
                                    f"{[(o.piece_config, o.x, o.y) for o in board.objects]}")
                break  # try again from beginning
    target_piece = board.objects[0]
    assert target_piece.piece_config == target_symbol
    task = Task(grid_config=board.grid_config, pieces=list(board.objects), target_piece=target_piece)
    return task


utterance_types = [
    [PropertyNames.COLOR],
    [PropertyNames.COLOR, PropertyNames.SHAPE],
    [PropertyNames.COLOR, PropertyNames.REL_POSITION],
    [PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION],
    [PropertyNames.SHAPE],
    [PropertyNames.SHAPE, PropertyNames.REL_POSITION],
    [PropertyNames.REL_POSITION],
]


def main(seed):
    assert seed is not None

    set_random_seed(seed)
    pieces_per_split = load_splits_from_json()

    for n, v in pieces_per_split.items():
        print(n, len(v))

    # when we start randomly, then worst case distance is double map size + some room for error
    sizes_and_pieces = {12: (2, 4, 30), 21: (4, 8, 60), 27: (6, 16, 80)}
    num_utts = 7
    for map_size, (max_pieces_per_pos, max_pieces, max_steps) in sizes_and_pieces.items():
        tasks = dict()
        for split_name, split_symbols in pieces_per_split.items():
            retries = 0
            target_symbols = split_symbols
            tasks[split_name] = []
            print("Generate for", split_name, map_size, "...")
            progress = tqdm(total=len(target_symbols) * num_utts)
            for target_symbol in target_symbols:
                if split_name == "train":
                    distractor_symbols = split_symbols
                else:
                    training_pieces = pieces_per_split["train"]
                    distractor_symbols = split_symbols + training_pieces
                sampler = UtteranceTypeOrientedDistractorSetSampler(distractor_symbols, target_symbol)
                num_pieces = random.randint(4, max_pieces)  # at least 4 pieces necessary
                for utterance_type in utterance_types:
                    success = False
                    group_sample_counter = 0
                    while not success:  # resample distractors if necessary
                        distractors_groups = sampler.sample_many_distractor_groups(utterance_type,
                                                                                   1,
                                                                                   (num_pieces, num_pieces),
                                                                                   pieces_per_pos=max_pieces_per_pos,
                                                                                   verbose=True)
                        for distractors_group in distractors_groups:  # only 1
                            try:
                                task = create_training_task(target_symbol, distractors_group, map_size, retries=100)
                                task.max_steps = max_steps
                                tasks[split_name].append(task)  # no extra targets !
                                success = True
                                retries = 0
                            except Exception as e:
                                print(f"({retries}) Re-try distractor group sampling")
                                retries += 1
                                group_sample_counter += 1
                                if group_sample_counter > 5:
                                    for p in [target_symbol] + distractors_group.pieces:
                                        print(f"- {p}")
                                    print(e)
                                    raise Exception("Too many re-tries for distractor groups")
                    progress.update(1)  # this counts the boards
            progress.close()
            print("Errors:", retries)
            print()
            # check_problems(tasks, split_name)
            # print_check(tasks, split_name)

        fn_tasks = f"tasks-didact-{map_size}.json"
        print(store_tasks_to_json(tasks, file_name=fn_tasks))


if __name__ == '__main__':
    seed = 42
    print(seed)
    main(seed=seed)
