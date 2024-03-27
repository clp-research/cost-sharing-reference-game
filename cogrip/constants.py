from cogrip.pentomino.symbolic.types import Colors, Shapes, RelPositions, Rotations

COLORS = list(Colors)
SHAPES = list(Shapes)
POSITIONS = list(RelPositions)
ROTATIONS = list(Rotations)

COLORS_6 = [Colors.RED, Colors.GREEN, Colors.BLUE, Colors.YELLOW, Colors.BROWN, Colors.PURPLE]
SHAPES_6 = [Shapes.P, Shapes.X, Shapes.T, Shapes.Z, Shapes.W, Shapes.U]
SHAPES_7 = [Shapes.P, Shapes.X, Shapes.T, Shapes.Z, Shapes.W, Shapes.U, Shapes.F]  # 5x5 kernels (no N or Y)
SHAPES_8 = [Shapes.P, Shapes.X, Shapes.T, Shapes.Z, Shapes.W, Shapes.U, Shapes.F, Shapes.V]  # 5x5 kernels
SHAPES_9 = [Shapes.P, Shapes.X, Shapes.T, Shapes.Z, Shapes.W, Shapes.U, Shapes.N, Shapes.F, Shapes.Y]  # >5x5
POSITIONS_8 = [RelPositions.TOP_LEFT, RelPositions.TOP_CENTER, RelPositions.TOP_RIGHT,
               RelPositions.RIGHT_CENTER, RelPositions.BOTTOM_RIGHT, RelPositions.BOTTOM_CENTER,
               RelPositions.BOTTOM_LEFT, RelPositions.LEFT_CENTER]  # no center
ORIENTATIONS = {
    Shapes.P: Rotations.DEGREE_0,
    Shapes.X: Rotations.DEGREE_0,
    Shapes.T: Rotations.DEGREE_0,
    Shapes.Z: Rotations.DEGREE_0,
    Shapes.W: Rotations.DEGREE_0,
    Shapes.U: Rotations.DEGREE_0,
    Shapes.N: Rotations.DEGREE_0,
    Shapes.F: Rotations.DEGREE_0,
    Shapes.Y: Rotations.DEGREE_0,
    Shapes.V: Rotations.DEGREE_0
}

# use 0 for "out-of-world object" positions
# use 1 for "no object" positions
COLOR_NAME_TO_IDX = dict((cn, idx) for cn, idx in zip([c.value_name for c in COLORS], range(2, len(COLORS) + 2)))
SHAPE_NAME_TO_IDX = dict((sn, idx) for sn, idx in zip([s.value for s in SHAPES], range(2, len(SHAPES) + 2)))
# for positions "no object" does not matter (thus only offset of 1 for out of world)
POS_NAME_TO_IDX = dict((pn, idx) for pn, idx in zip([p.name for p in POSITIONS], range(1, len(POSITIONS) + 1)))

IDX_TO_POS = dict((idx, p) for p, idx in zip([p for p in POSITIONS], range(1, len(POSITIONS) + 1)))
IDX_TO_POS[0] = None  # oow

IDX_TO_COLOR_NAME = dict((idx, cn) for cn, idx in zip([c.value_name for c in COLORS], range(2, len(COLORS) + 2)))
IDX_TO_COLOR_NAME[0] = "oow"
IDX_TO_COLOR_NAME[1] = "empty"

IDX_TO_SHAPE_NAME = dict((idx, sn) for sn, idx in zip([s.value for s in SHAPES], range(2, len(SHAPES) + 2)))
IDX_TO_SHAPE_NAME[0] = "oow"
IDX_TO_SHAPE_NAME[1] = "empty"
