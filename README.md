# Blunder 0 (BC0)

**Blunder 0**, also known as **BC0**, is a UCI-compatible chess engine written in Python. It searches positions with iterative deepening, negamax, alpha-beta pruning, quiescence search, and a transposition table. Its evaluator can use either a trained NumPy neural network or a built-in material and positional evaluation.

## Features

- UCI support for chess GUIs such as Arena, Cute Chess, and Banksia
- Iterative deepening with principal-variation search and aspiration windows
- Alpha-beta pruning, late-move reductions, and guarded null-move pruning
- Quiescence search, move ordering, killer moves, and history heuristics
- Transposition table that persists across moves in a game
- Neural evaluator training from FEN and evaluation data
- Built-in fallback evaluation when no model is loaded

## Requirements

- Python 3.10 or newer
- Dependencies listed in [`requirements.txt`](requirements.txt)

Install the dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run in a chess GUI

Start BC0 in UCI mode:

```powershell
python -m chess_engine
```

In your chess GUI, add a UCI engine with your Python interpreter as the executable and `-m chess_engine` as the argument. The engine identifies itself as **Blunder 0**. Configure the `Depth` and `Hash` options in the GUI. To load a trained evaluator, set `EvalFile` to its full path.

You can also start the engine with a default search depth:

```powershell
python -m chess_engine uci --depth 10
```

The GUI can override the default depth with `go depth`, or use `go movetime` and clock commands. Search depth can be set from 1 to 30. Higher depths can take substantially longer.

## Train a neural evaluator

Training data must be a CSV containing `FEN` and `evaluation` columns. Evaluations should be in pawn units from White's point of view. If your data is in centipawns, divide the scores by 100 before training.

Train with the default settings:

```powershell
python -m chess_engine train evaluated.csv --output models\evaluator.npz
```

The trainer reads the CSV in chunks rather than loading all features into memory. Defaults are 5 epochs, batch size 256, hidden width 64, and 4 BLAS threads. You can adjust these settings:

```powershell
python -m chess_engine train evaluated.csv `
  --output models\evaluator.npz `
  --epochs 5 --batch-size 256 --hidden 64 --threads 4
```

Other options include `--learning-rate`, `--limit`, `--seed`, and `--chunk-size`. Use `--limit 100000` to train on at most the first 100,000 CSV rows; omit it to use all rows. Set the UCI `EvalFile` option to the generated model to use it during play.

## Build a Windows executable

With dependencies installed, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_exe.ps1
```

The executable is created at `dist\ChessEngine.exe`. The evaluator model remains a separate file and can be selected through `EvalFile`.

## Project layout

```text
chess_engine/
├── __main__.py   # Command-line entry point
├── features.py   # Position encoding shared by training and evaluation
├── network.py    # NumPy neural evaluator and trainer
├── search.py     # Position evaluation and search
└── uci.py        # UCI protocol adapter
engine.py         # Compatibility entry point
requirements.txt
build_exe.ps1
```

## Limitations

BC0 is a compact, CPU-oriented engine and is not at Stockfish strength. Its playing strength depends on search time and, when used, the quality and score conventions of the training data.
