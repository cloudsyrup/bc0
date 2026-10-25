import shlex
import sys
import threading
import time
import chess
from .search import Search, MATE
from .network import Evaluator


class UCI:
    def __init__(self):
        self.depth, self.hash_mb = 8, 64
        self.model_path = ""
        self.evaluator = None
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.worker = None
        self.searcher = None
        self.started = time.monotonic()

    def send(self, line):
        with self.lock:
            print(line, flush=True)

    def info(self, depth, score, nodes, move):
        elapsed = max(1, int((time.monotonic() - self.started) * 1000))
        if abs(score) >= MATE - 100:
            mate = (MATE - abs(score) + 1) // 2
            score_text = f"mate {mate if score > 0 else -mate}"
        else:
            score_text = f"cp {score}"
        self.send(f"info depth {depth} score {score_text} nodes {nodes} nps {nodes * 1000 // elapsed} time {elapsed} pv {move.uci()}")

    def start_go(self, args):
        if self.worker and self.worker.is_alive(): self.stop()
        opts = {args[i]: args[i + 1] for i in range(len(args) - 1)
                if args[i] in ("depth", "movetime", "wtime", "btime", "winc", "binc")}
        depth = 30 if "infinite" in args else int(opts.get("depth", self.depth))
        mt = int(opts["movetime"]) / 1000 if "movetime" in opts else None
        if mt is None:
            clock_key = "wtime" if self.board.turn else "btime"
            inc_key = "winc" if self.board.turn else "binc"
            if clock_key in opts:
                remaining = int(opts[clock_key]) / 1000
                inc = int(opts.get(inc_key, 0)) / 1000
                mt = max(.05, min(remaining * .7, remaining / 25 + inc * .8))
        board = self.board.copy()
        self.started = time.monotonic()
        if self.searcher is None:
            self.searcher = Search(self.evaluator, self.hash_mb, self.info)
        else:
            self.searcher.evaluator = self.evaluator
            self.searcher.tt_limit = max(10000, self.hash_mb * 16000)
        def work():
            move, _ = self.searcher.run(board, depth, mt)
            self.send(f"bestmove {move.uci() if move else '0000'}")
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self):
        if self.searcher: self.searcher.stop = True
        if self.worker and self.worker.is_alive(): self.worker.join()

    def setoption(self, line):
        try:
            parts = shlex.split(line)
            name_at = parts.index("name") + 1
            value_at = parts.index("value") if "value" in parts else len(parts)
            name = " ".join(parts[name_at:value_at]).lower()
            value = " ".join(parts[value_at + 1:]) if value_at < len(parts) else ""
            if name == "depth": self.depth = max(1, min(30, int(value)))
            elif name == "hash": self.hash_mb = max(1, min(4096, int(value)))
            elif name == "evalfile":
                self.model_path = value
                self.evaluator = Evaluator(value) if value else None
            elif name == "clear hash":
                if self.searcher: self.searcher.tt.clear()
        except Exception as e:
            self.send(f"info string setoption failed: {e}")

    def position(self, args):
        try:
            if args[0] == "startpos":
                board, idx = chess.Board(), 1
            elif args[0] == "fen":
                mi = args.index("moves") if "moves" in args else len(args)
                board, idx = chess.Board(" ".join(args[1:mi])), mi
            else: return
            if idx < len(args) and args[idx] == "moves":
                for token in args[idx + 1:]: board.push_uci(token)
            self.board = board
        except Exception as e:
            self.send(f"info string invalid position: {e}")

    def loop(self):
        self.send("id name Blunder 0")
        self.send("id author Shubham")
        self.send("option name Depth type spin default 8 min 1 max 30")
        self.send("option name Hash type spin default 64 min 1 max 4096")
        self.send("option name EvalFile type string default")
        self.send("option name Clear Hash type button")
        self.send("uciok")
        for raw in sys.stdin:
            line = raw.strip()
            if not line: continue
            cmd, _, tail = line.partition(" ")
            if cmd == "isready": self.send("readyok")
            elif cmd == "setoption": self.setoption(tail)
            elif cmd == "ucinewgame":
                self.stop(); self.board = chess.Board()
                if self.searcher: self.searcher.tt.clear()
            elif cmd == "position": self.position(tail.split())
            elif cmd == "go": self.start_go(tail.split())
            elif cmd == "stop": self.stop()
            elif cmd == "quit": self.stop(); break
            elif cmd == "debug": pass


def main():
    UCI().loop()
