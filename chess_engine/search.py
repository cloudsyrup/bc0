import time
import chess
import chess.polyglot

MATE = 100000
INF = 200000
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


class Search:
    def __init__(self, evaluator=None, hash_mb=64, info=None):
        self.evaluator, self.info = evaluator, info
        self.tt_limit = max(10000, hash_mb * 16000)
        self.tt = {}
        self.nodes = 0
        self.stop = False
        self.deadline = None
        self.killers = {}
        self.history = {}

    def evaluate(self, board):
        if self.evaluator:
            white_score = self.evaluator.predict_board(board) * 100
        else:
            white_score = 0
            pawn_squares = {
                chess.WHITE: board.pieces(chess.PAWN, chess.WHITE),
                chess.BLACK: board.pieces(chess.PAWN, chess.BLACK),
            }
            for sq, piece in board.piece_map().items():
                sign = 1 if piece.color else -1
                rank = chess.square_rank(sq) if piece.color else 7 - chess.square_rank(sq)
                center = 7 - (abs(3.5 - chess.square_file(sq)) + abs(3.5 - rank))
                bonus = 0
                if piece.piece_type == chess.PAWN: bonus = rank * 7 + center * 2
                elif piece.piece_type in (chess.KNIGHT, chess.BISHOP): bonus = center * 7
                elif piece.piece_type == chess.KING and board.fullmove_number > 30: bonus = center * 4
                white_score += sign * (VALUES[piece.piece_type] + bonus)
            # Reward passed pawns and bishop pairs; penalize doubled pawns.
            for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
                own = pawn_squares[color]
                enemy = pawn_squares[not color]
                file_counts = {}
                for sq in own:
                    file_ = chess.square_file(sq)
                    file_counts[file_] = file_counts.get(file_, 0) + 1
                white_score -= sign * sum(max(0, n - 1) * 12 for n in file_counts.values())
                for sq in own:
                    file_, rank = chess.square_file(sq), chess.square_rank(sq)
                    forward_enemy = [esq for esq in enemy if
                                     (chess.square_rank(esq) > rank if color else
                                      chess.square_rank(esq) < rank)]
                    if not any(abs(chess.square_file(esq) - file_) <= 1 for esq in forward_enemy):
                        relative_rank = rank if color else 7 - rank
                        white_score += sign * (12 + relative_rank * 8)
                if len(board.pieces(chess.BISHOP, color)) >= 2:
                    white_score += sign * 28
        return int(white_score if board.turn else -white_score)

    def _draw(self, board, ply):
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return True
        # Repetition scanning is proportional to the move stack. Twofold repetition
        # is a useful cycle-pruning heuristic inside the tree; check official threefold
        # claims only at the root.
        if board.move_stack:
            return board.is_repetition(3 if ply == 0 else 2)
        return False


    @staticmethod
    def _key(board):
        # The fifty-move clock changes whether a cached position is drawable.
        return (chess.polyglot.zobrist_hash(board), min(board.halfmove_clock, 100))

    def _check_stop(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop = True
        return self.stop

    def _ordered(self, board, tt_move=None, ply=0, moves=None):
        def score(move):
            if move == tt_move: return 10_000_000
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if board.is_en_passant(move): victim_value = VALUES[chess.PAWN]
                else: victim_value = VALUES[victim.piece_type] if victim else 0
                attacker = board.piece_at(move.from_square)
                return 1_000_000 + victim_value * 10 - (VALUES[attacker.piece_type] if attacker else 0)
            if move.promotion: return 900_000 + VALUES[move.promotion]
            if move in self.killers.get(ply, ()): return 800_000 - self.killers[ply].index(move)
            return self.history.get((board.turn, move.from_square, move.to_square), 0)
        return sorted(board.legal_moves if moves is None else moves, key=score, reverse=True)

    def _qsearch(self, board, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 255) == 0 and self._check_stop(): return self.evaluate(board)
        if self._draw(board, ply): return 0
        if board.is_check():
            best = -INF
            moves = list(board.legal_moves)
            if not moves: return -MATE + ply
        else:
            best = self.evaluate(board)
            if best >= beta:
                return 0 if board.is_stalemate() else beta
            alpha = max(alpha, best)
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
            if not moves and board.is_stalemate(): return 0

        for move in self._ordered(board, ply=ply, moves=moves):
            board.push(move); score = -self._qsearch(board, -beta, -alpha, ply + 1); board.pop()
            if self.stop: return best
            if score > best: best = score
            if score >= beta: return beta
            alpha = max(alpha, score)
        return best

    def _search(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 255) == 0 and self._check_stop(): return self.evaluate(board)
        if self._draw(board, ply): return 0
        if board.is_checkmate(): return -MATE + ply
        if depth <= 0: return self._qsearch(board, alpha, beta, ply)
        original_alpha = alpha
        key = self._key(board)
        entry = self.tt.get(key)
        tt_move = None
        if entry:
            td, ts, bound, tm = entry
            ts = self._from_tt(ts, ply)

            tt_move = tm
            if td >= depth:
                if bound == 0: return ts
                if bound == 1 and ts >= beta: return ts
                if bound == 2 and ts <= alpha: return ts
        best, best_move = -INF, None
        in_check = board.is_check()
        # Null-move pruning cuts branches when even passing the turn is sufficient
        # to exceed beta. Disable it in check and in pawn-only endings (zugzwang risk).
        if (depth >= 3 and ply > 0 and not in_check and
                (not board.move_stack or not board.peek().is_null()) and beta < MATE - 1000 and
                any(board.pieces(pt, board.turn) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))):
            reduction = 2 + depth // 6
            board.push(chess.Move.null())
            null_score = -self._search(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
            board.pop()
            if self.stop: return null_score
            if null_score >= beta: return null_score

        moves = self._ordered(board, tt_move, ply)
        if not moves: return -MATE + ply if board.is_check() else 0
        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            board.push(move)
            child_depth = depth - 1
            # Late-move reductions: spend less effort on quiet moves ordered late,
            # then restore the full search if the reduced search improves alpha.
            reduction = 1 if depth >= 3 and move_index >= 4 and not is_capture and not gives_check and not in_check else 0
            if move_index == 0:
                score = -self._search(board, child_depth, -beta, -alpha, ply + 1)
            else:
                score = -self._search(board, child_depth - reduction, -alpha - 1, -alpha, ply + 1)
                if score > alpha and (reduction or score < beta):
                    score = -self._search(board, child_depth, -beta, -alpha, ply + 1)
            board.pop()
            if self.stop: return best if best_move else score
            if score > best: best, best_move = score, move
            if score > alpha: alpha = score
            if alpha >= beta:
                if not is_capture:
                    killers = self.killers.setdefault(ply, [])
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    k = (board.turn, move.from_square, move.to_square)
                    bonus = depth * depth
                    old = self.history.get(k, 0)
                    self.history[k] = max(-16384, min(16384, old + bonus - old * bonus // 16384))
                break
        bound = 2 if best <= original_alpha else (1 if best >= beta else 0)
        if len(self.tt) >= self.tt_limit: self.tt.clear()
        self.tt[key] = (depth, self._to_tt(best, ply), bound, best_move)
        return best

    @staticmethod
    def _to_tt(score, ply):
        if score > MATE - 1000: return score + ply
        if score < -MATE + 1000: return score - ply
        return score

    @staticmethod
    def _from_tt(score, ply):
        if score > MATE - 1000: return score - ply
        if score < -MATE + 1000: return score + ply
        return score


    def run(self, board, depth=8, movetime=None):
        self.nodes, self.stop, self.killers, self.history = 0, False, {}, {}
        self.deadline = time.monotonic() + movetime if movetime else None
        best_move = next(iter(board.legal_moves), None)
        if best_move is None: return None, 0
        best_score = 0
        for d in range(1, max(1, depth) + 1):
            window = 35 if d >= 3 else INF
            alpha, beta = max(-INF, best_score - window), min(INF, best_score + window)
            score = self._search(board, d, alpha, beta, 0)
            while not self.stop and (score <= alpha or score >= beta) and window < INF:
                window *= 2
                alpha, beta = max(-INF, best_score - window), min(INF, best_score + window)
                score = self._search(board, d, alpha, beta, 0)
            if self.stop: break
            entry = self.tt.get(self._key(board))
            if entry and entry[3] is not None: best_move = entry[3]
            best_score = score
            if self.info:
                self.info(d, score, self.nodes, best_move)
            if abs(score) >= MATE - 100: break
        return best_move, best_score
