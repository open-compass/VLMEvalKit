from vlmeval.dataset.utils.mmhelix.evaluator import *
from vlmeval.dataset.utils.mmhelix.evaluators.graph_problems_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.twentyfourpoints_evaluator import *
from vlmeval.dataset.utils.mmhelix.evaluators.calcudoku_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.cryptomath_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.kukurasu_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.skyscrapers_evaluator import *
from vlmeval.dataset.utils.mmhelix.evaluators.wordladder_eval import *

from vlmeval.dataset.utils.mmhelix.evaluators.aquarium_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.binario_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.campsite_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.eulero_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.futoshiki_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.hanoi_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.hitori_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.nonogram_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.bridges_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.kakuro_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.maze_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.minesweeper_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.nibbles_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.numbrix_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.slidingpuzzle_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.sokoban_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.snake_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.wordsearch_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.shingoki_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.tapa_eval import *
from vlmeval.dataset.utils.mmhelix.evaluators.sudoku_evaluator import *

metrics = {
    'simple_str_match': SimpleStrMatch(),
    'match_from_list': MatchFromList(),
    'sliding_puzzle_evaluator': SlidingPuzzleEvaluator(),
    'eulero_evaluator': EuleroEvaluator(),
    'hanoi_evaluator': TowerOfHanoiEvaluator(),
    'maze_evaluator': MazeEvaluator(),
    'minesweeper_evaluator': MinesweeperEvaluator(),
    'numbrix_evaluator': NumbrixEvaluator(),
    'sokoban_evaluator': SokobanEvaluator(),
    'snake_evaluator': SnakeEvaluator(),
    'wordsearch_evaluator': WordSearchEvaluator(),
    'hamiltonian_path_evaluator': HamiltonianPathEvaluator(),
    'hamiltonian_cycle_evaluator': HamiltonianCycleEvaluator(),
    'eulerian_path_evaluator': EulerianPathEvaluator(),
    'eulerian_cycle_evaluator': EulerianCycleEvaluator(),
    'topological_sort_evaluator': TopologicalSortEvaluator(),
    '24points_evaluator': TwentyFourPointsEvaluator(),
    'calcudoku_evaluator': CalcudokuEvaluator(),
    'cryptomath_evaluator': CryptoMathEvaluator(),
    'kukurasu_evaluator': KukurasuEvaluator(),
    'skyscrapers_evaluator': SkyscrapersEvaluator(),
    'wordladder_evaluator': WordLadderEvaluator(),
    'aquarium_evaluator': AquariumEvaluator(),
    'binairo_evaluator': BinarioEvaluator(),
    'campsite_evaluator': CampsiteEvaluator(),
    'futoshiki_evaluator': FutoshikiEvaluator(),
    'hitori_evaluator': HitoriEvaluator(),
    'nonogram_evaluator': NonogramsEvaluator(),
    'bridges_evaluator': BridgesEvaluator(),
    'kakuro_evaluator': KakuroEvaluator(),
    'shingoki_evaluator': ShingokiEvaluator(),
    'tapa_evaluator': TapaEvaluator(),
    'nibbles_evaluator': NibblesEvaluator(),
    'connectivity_evaluator': ConnectivityEvaluator(),
    'sudoku_evaluator': SudokuEvaluator(),
    'unsupported': SimpleStrMatch(),
}
