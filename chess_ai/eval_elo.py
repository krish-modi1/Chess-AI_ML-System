"""
eval_elo.py — Play ChessCNN vs Stockfish, collect PGN, run BayesElo.

Usage:
    python chess_ai/eval_elo.py \
        --model  chess_ai/game_engine/model/best_model.pth \
        --stockfish /usr/games/stockfish \
        --games 50 --elo 1320 --sims 400

Outputs:
    eval_games.pgn          — all games in standard PGN
    bayeselo_results.txt    — BayesElo rating output
"""

import argparse, datetime, os, sys, warnings
warnings.filterwarnings('ignore', category=UserWarning)
import chess, chess.pgn, chess.engine
import torch
from tqdm import tqdm

# chess_ai/ on path so `game_engine` resolves as a package;
# game_engine/ on path so the mcts_engine_cpp .so is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'game_engine'))

from game_engine.cnn import ChessCNN                        # type: ignore[import-untyped]
from game_engine.chess_env import ChessGame                 # type: ignore[import-untyped]
from game_engine.mcts_worker_cpp import MCTSWorker          # type: ignore[import-untyped]
from game_engine.bayeselo_runner import BayesEloRunner      # type: ignore[import-untyped]


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(path, device):
    model = ChessCNN()
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    # torch.compile() wraps keys with '_orig_mod.' — strip it for inference
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(model, worker, engine, stockfish_elo, model_is_white, move_time):
    game_state = ChessGame()
    history    = []   # list[chess.Board], most recent first, up to 7

    pgn = chess.pgn.Game()
    pgn.headers['Date']  = datetime.date.today().strftime('%Y.%m.%d')
    pgn.headers['White'] = 'Model'     if model_is_white else 'Stockfish'
    pgn.headers['Black'] = 'Stockfish' if model_is_white else 'Model'
    node = pgn

    while not game_state.is_over:
        model_turn = (game_state.board.turn == chess.WHITE) == model_is_white

        if model_turn:
            move_uci, _ = worker.search_direct(
                game_state, model, temperature=0.1, history=history
            )
        else:
            result   = engine.play(game_state.board, chess.engine.Limit(time=move_time))
            move_uci = result.move.uci()

        worker.advance_root(move_uci)
        history.insert(0, game_state.board.copy())
        if len(history) > 7:
            history.pop()

        game_state.push(move_uci)
        node = node.add_variation(chess.Move.from_uci(move_uci))

    worker.reset_cache()
    result_str = game_state.result
    pgn.headers['Result'] = result_str
    return result_str, pgn


# ── BayesElo ─────────────────────────────────────────────────────────────────

def run_bayeselo(bayeselo_bin, pgn_path, stockfish_elo):
    runner = BayesEloRunner(
        project_root=os.path.dirname(os.path.abspath(__file__)),
        stockfish_elo=stockfish_elo,
    )
    runner.bayeselo_path = bayeselo_bin  # use the resolved path from args
    result = runner.run(pgn_path)
    if result:
        ci = (result['model_ci_upper'] - result['model_ci_lower']) / 2
        print(f'\n── BayesElo result ──────────────────────────────────')
        print(f'  ChessCNN ELO : {result["model_elo"]:.0f} ± {ci:.0f}')
        print(f'  vs Stockfish : {stockfish_elo}')
        print(f'  Difference   : {result["diff_elo"]:+.0f} ELO')
        print(f'  95% CI       : [{result["diff_ci_lower"]:.0f}, {result["diff_ci_upper"]:.0f}]')
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',     required=True,            help='Path to best_model.pth')
    p.add_argument('--stockfish', default='stockfish',      help='Stockfish binary')
    _default_be = os.path.join(os.path.dirname(__file__), 'BayesElo', 'bayeselo')
    p.add_argument('--bayeselo',  default=_default_be,      help='BayesElo binary')
    p.add_argument('--games',     type=int,   default=50,   help='Total games (split evenly per colour)')
    p.add_argument('--elo',       type=int,   default=1320, help='Stockfish UCI_Elo target')
    p.add_argument('--sims',      type=int,   default=400,  help='MCTS simulations per move')
    p.add_argument('--batch',     type=int,   default=8,    help='MCTS leaf batch size')
    p.add_argument('--move-time', type=float, default=0.1,  help='Stockfish seconds per move')
    p.add_argument('--out',       default='eval_games.pgn', help='Output PGN path')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')
    print(f'Model   : {args.model}')
    print(f'Games   : {args.games}  ({args.games//2} as White, {args.games - args.games//2} as Black)')
    print(f'Stockfish ELO: {args.elo}  |  MCTS sims: {args.sims}')

    model  = load_model(args.model, device)
    worker = MCTSWorker(worker_id=0, input_queue=None, output_queue=None,
                        simulations=args.sims, batch_size=args.batch)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': args.elo})

    wins = draws = losses = 0
    colour_schedule = (
        [True]  * (args.games // 2) +
        [False] * (args.games - args.games // 2)
    )

    pgn_path = args.out
    results_path = pgn_path.replace('.pgn', '_bayeselo.txt')

    with open(pgn_path, 'w') as pgn_file:
        for game_idx, model_is_white in enumerate(tqdm(colour_schedule, desc='Games'), 1):
            result_str, pgn = play_game(
                model, worker, engine, args.elo, model_is_white, args.move_time
            )

            print(pgn, file=pgn_file, end='\n\n')
            pgn_file.flush()

            model_colour = 'White' if model_is_white else 'Black'
            if result_str == '1/2-1/2':
                draws += 1; tag = '½'
            elif (result_str == '1-0') == model_is_white:
                wins  += 1; tag = '✓'
            else:
                losses += 1; tag = '✗'

            tqdm.write(f'  Game {game_idx:3d} [{model_colour:5s}]  {result_str}  {tag}  '
                       f'W={wins} D={draws} L={losses}')

    engine.quit()

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total
    print(f'\nResult: {wins}W / {draws}D / {losses}L  score={score:.3f}')
    print(f'PGN saved: {pgn_path}')

    # Run BayesElo if binary is available
    if os.path.isfile(args.bayeselo):
        os.chmod(args.bayeselo, 0o755)
        run_bayeselo(args.bayeselo, pgn_path, args.elo)
    else:
        print(f'\nBayesElo binary not found: {args.bayeselo}')
        print('Expected at chess_ai/BayesElo/bayeselo (already in repo).')


if __name__ == '__main__':
    main()
