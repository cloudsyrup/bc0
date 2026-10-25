import re
import pandas as pd
import chess

# Input and output files
INPUT_CSV = "games.csv"
OUTPUT_CSV = "positions_evaluations.csv"


def extract_positions_and_evals(moves_text):
    """
    Extract the FEN after every chess move and its [%eval] value.
    FEN represents the position AFTER the move is played.
    """
    if pd.isna(moves_text) or not str(moves_text).strip():
        return []

    board = chess.Board()
    results = []

    # Split the move text into comments and individual tokens
    tokens = re.findall(r"\{[^}]*\}|[^\s]+", str(moves_text))

    for token in tokens:

        # Extract evaluation from a PGN comment, e.g.:
        # {[%eval 0.18] [%clk 0:10:00]}
        if token.startswith("{") and token.endswith("}"):
            match = re.search(
                r"\[%eval\s+([^\]\s]+)\]",
                token
            )

            if match and results:
                results[-1]["eval"] = match.group(1)

            continue

        # Skip standalone comments, game results and NAGs
        if token.startswith("$") or token in {
            "1-0", "0-1", "1/2-1/2", "*"
        }:
            continue

        # Remove move-number prefixes, e.g. 1.e4 or 12...Nf6
        token = re.sub(r"^\d+\.(?:\.\.)?", "", token)

        if not token:
            continue

        # Skip remaining move numbers
        if re.fullmatch(r"\d+\.(?:\.\.)?", token):
            continue

        # Remove chess annotation symbols, e.g. a4?! or Nxe4??
        san = re.sub(r"[!?]+$", "", token)

        try:
            # Parse the move in the current board position
            move = board.parse_san(san)

            # Play the move
            board.push(move)

            # Store the resulting position in FEN format
            results.append({
                "position": board.fen(),
                "eval": ""
            })

        except ValueError:
            # Ignore tokens that are not valid moves
            print(f"Skipping unrecognized move: {token}")

    return results


# Load the source CSV
df = pd.read_csv(INPUT_CSV)

if "Moves" not in df.columns:
    raise ValueError("The input CSV must contain a 'Moves' column.")

all_rows = []

# Process each game independently, starting from the initial position
for game_moves in df["Moves"]:
    all_rows.extend(extract_positions_and_evals(game_moves))

# Create a CSV with exactly two columns
output_df = pd.DataFrame(
    all_rows,
    columns=["position", "eval"]
)

output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done! Extracted {len(output_df)} positions.")
print(f"Saved to: {OUTPUT_CSV}")
print(output_df.head())
