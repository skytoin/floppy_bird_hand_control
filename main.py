"""Entry point for the hand-controlled Floppy Bird game."""

from __future__ import annotations

import argparse

from game import FloppyBirdGame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Floppy Bird with hand flaps!")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Disable the webcam hand controller and fall back to keyboard controls.",
    )
    parser.add_argument(
        "--debug-hand",
        action="store_true",
        help="Show a debug window with the MediaPipe hand landmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game = FloppyBirdGame(enable_hand_control=not args.no_hand, debug_hand=args.debug_hand)
    game.run()


if __name__ == "__main__":
    main()
