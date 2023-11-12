"""
The Brandeis Quant Club ML/AI Competition (November 2023)

Author: @Ephraim Zimmerman
Email: quants@brandeis.edu
Website: brandeisquantclub.com; quants.devpost.com

Description:

For any technical issues or questions please feel free to reach out to
the "on-call" hackathon support member via email at quants@brandeis.edu

Website/GitHub Repository:
You can find the latest updates, documentation, and additional resources for this project on the
official website or GitHub repository: https://github.com/EphraimJZimmerman/chess_hackathon_23

License:
This code is open-source and released under the MIT License. See the LICENSE file for details.
"""

import random
import chess
import time
from collections.abc import Iterator
from contextlib import contextmanager
import test_bot
import numpy as np
import pandas as pd
import chess.pgn
import os
import io
import zstandard
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


@contextmanager
def game_manager() -> Iterator[None]:
    """Creates context for the game."""
    print("===== GAME STARTED =====")
    ping: float = time.perf_counter()
    try:
        # DO NOT EDIT. This will be replaced w/ judging context manager.
        yield
    finally:
        pong: float = time.perf_counter()
        total = pong - ping
        print(f"Total game time = {total:.3f} seconds")
    print("===== GAME ENDED =====")


class Bot:
    def __init__(self, fen=None):
        self.board = chess.Board(fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def check_move_is_legal(self, initial_position, new_position) -> bool:
        """
        To check if, from an initial position, the new position is valid.

        Args:
            initial_position (str): The starting position given chess notation.
            new_position (str): The new position given chess notation.

        Returns:
            bool: If this move is legal
        """
        return chess.Move.from_uci(initial_position + new_position) in self.board.legal_moves

    def next_move(self) -> str:
        """
        The main call and response loop for playing a game of chess.

        Returns:
            str: The current location and the next move.
        """
        scholars_mate_moves = ['e2e4', 'e7e5', 'd1h5', 'g8f6', 'f1c4']

        if len(self.board.move_stack) < len(scholars_mate_moves):
            move = scholars_mate_moves[len(self.board.move_stack)]
            print("My move: " + move)
            return move

        # If Scholar's Mate is complete, return a random legal move
        move = str(random.choice([_ for _ in self.board.legal_moves]))
        print("My move: " + move)
        return move

    def is_scholars_mate(self, board):
        """
        Checks if the current board position is Scholar's Mate.

        Args:
            board (chess.Board): The chess board.

        Returns:
            bool: True if it's Scholar's Mate, False otherwise.
        """
        scholars_mate_moves = ['e2e4', 'e7e5', 'd1h5', 'g8f6', 'f1c4']

        temp_board = board.copy()  # Make a copy of the board to avoid modifying the original

        for move in scholars_mate_moves:
            uci_move = chess.Move.from_uci(move)
            if uci_move not in temp_board.legal_moves:
                return False
            temp_board.push(uci_move)

        return temp_board.is_checkmate() and temp_board.fullmove_number == len(scholars_mate_moves) // 2


# Add promotion stuff
if __name__ == "__main__":
    chess_bot = Bot()

    with game_manager():
        playing = True

        while playing:
            if chess_bot.is_scholars_mate(chess_bot.board):
                print("Scholars Mate!")
                playing = False
                break

            if chess_bot.board.turn:
                move = test_bot.get_move(chess_bot.board)
            else:
                move = chess_bot.next_move()

            print("Move:", move)

            if chess_bot.check_move_is_legal(move[:2], move[2:]):
                chess_bot.board.push_uci(move)
            else:
                print("Illegal move! Ending the game.")
                playing = False

            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")
                elif chess_bot.is_scholars_mate(chess_bot.board):
                    print("Scholars Mate!")

                print(chess_bot.board.outcome())

                playing = False
