# Chess Neural Search Engine

A UCI chess engine written in Python. It combines a neural position evaluator trained on FEN and evaluation with iterative deepening, negamax minimax, alpha-beta pruning, quiescence search, move ordering, and a transposition table. UCI support lets standard chess GUI programs launch it as an external engine.

## Project layout

```text
Chess/
├── evaluated.csv             # supplied training data
├── chess_engine/
│   ├── __main__.py            # command line entry point
│   ├── features.py            # position encoding used by training and engine
│   ├── network.py             # NumPy neural model and trainer
│   ├── search.py              # minimax search and fallback evaluator
│   └── uci.py                 # GUI protocol
├── models/                    # trained evaluator is saved here
├── requirements.txt
└── build_exe.ps1
```

## Install and train

Use Python 3.10 or newer. From PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m chess_engine train evaluated.csv --output models\evaluator.npz
```

The default run uses **all rows** (the supplied CSV has 713,000 data rows), 5 epochs, batch size 256, hidden width 64, and 4 BLAS threads. It reads and shuffles 8,192-position chunks, then trains each chunk in mini-batches; it never loads the full feature dataset into RAM. These defaults leave CPU headroom on a 6-core processor. Use `--threads 2 --batch-size 128` to make it lighter, or `--threads 6 --batch-size 512` to use more CPU. For example, this lower-load command still includes all positions:

```powershell
python -m chess_engine train evaluated.csv --output models\evaluator.npz --threads 2 --batch-size 128
```

Training accepts `--epochs`, `--batch-size`, `--hidden`, `--learning-rate`, `--limit`, `--seed`, `--chunk-size`, and `--threads`. `--limit 100000` trains on only the first 100,000 rows; omit it (or use zero) to use the entire file. Chunk progress and each epoch's mean Huber loss are printed. The model is a two-hidden-layer tanh MLP trained with Adam. It reads the `FEN` and `evaluation` columns, skips invalid rows, clips extreme labels to +/-20, and assumes evaluations are in pawn units from White's point of view. If your labels use centipawns, divide them by 100 before training or the evaluator scale will be wrong.

The model input represents all piece/color/square occupancies, side to move, and castling rights. The same feature encoder is used at training and search time. If no trained model is loaded, the engine uses a material and piece-square evaluator.

## Run from a chess GUI

UCI mode is the default:

```powershell
python -m chess_engine
```

In a UCI-compatible GUI, add an engine and point its executable to the Python interpreter in `.venv\Scripts\python.exe`; set its arguments to `-m chess_engine`. In the engine options, set `EvalFile` to the full path of `models\evaluator.npz`. Set `Depth` (default 8) and `Hash` (transposition-table budget in MB) in the GUI. The engine also accepts standard `go depth`, `go movetime`, and clock commands from the GUI. Greater depth takes substantially longer; available search depth is configurable from 1 to 30.

You can start the UCI engine with a depth default from the command line:

```powershell
python -m chess_engine uci --depth 10
```

### Building the standalone Windows executable

With the virtual environment active and dependencies installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_exe.ps1
```

This creates `dist\ChessEngine.exe`. Add that executable to the GUI as a UCI engine. Set the GUI's `EvalFile` option to the model path (for example, `C:\path\to\Chess\models\evaluator.npz`). The model is intentionally a separate file so you can retrain it without rebuilding the executable. Build on Windows to create a Windows executable.

## Search outline

At each iteration the engine searches the root to the configured depth. Negamax evaluates from the side-to-move perspective; alpha-beta cuts branches that cannot change the result. Principal-variation search uses a narrow window after the first move, aspiration windows reuse the previous iteration's score, and late-move reductions search late quiet moves at reduced depth before re-searching any that improve alpha. Guarded null-move pruning cuts branches where passing the turn still exceeds beta; it is disabled in check and pawn-only endings. Quiescence search follows captures and promotions, and searches all legal evasions when in check. The engine orders transposition-table moves, captures by victim value, promotions, killer moves, and history-scored moves. Its transposition table persists across moves in one game and clears on `ucinewgame`. Iterative deepening reports the last fully completed result, so UCI `stop` can return a legal move while a deeper iteration is underway. Fifty-move, repetition, stalemate, and insufficient-material draws are recognized.

## Notes

- This is a compact CPU-oriented engine and baseline neural model, not a Stockfish-strength engine. Search quality depends on the labels, their units and perspective, model training, and available time.
- UCI scores are reported in centipawns; mate scores are reported in moves. Evaluation labels must use White's perspective as noted above.
- The `Depth` UCI option is the default for `go` commands without an explicit depth. A GUI may override it per search.
