import numpy as np
import chess

FEATURES = 12 * 64 + 5




def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(FEATURES, dtype=np.float32)
    for square, piece in board.piece_map().items():
        # Keep white and black on separate planes; mirror black ranks to share patterns.
        sq = square if piece.color == chess.WHITE else chess.square_mirror(square)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        x[plane * 64 + sq] = 1.0
    x[768] = 1.0 if board.turn == chess.WHITE else -1.0
    x[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[772] = float(board.has_queenside_castling_rights(chess.BLACK))
    return x
