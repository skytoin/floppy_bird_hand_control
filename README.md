# Floppy Bird Hand Control

A playful take on Flappy Bird where you flap your hand in front of a webcam to
make the bird soar. Built with Pygame for the game loop and MediaPipe Hands for
hand tracking.

## Features

- Hand tracking powered flap detection using MediaPipe Hands.
- Smooth Pygame implementation with pipes, score keeping, and game over state.
- Keyboard fallbacks so you can still play if no camera is available (press
  `SPACE` or `UP`).
- Optional debug window to visualize hand landmarks when fine-tuning the
  controller.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

On Linux you may need additional system packages for OpenCV and Pygame (SDL).

## Running the Game

Run the entry point:

```bash
python main.py
```

Useful flags:

- `--no-hand` — disable the webcam controller and use keyboard-only controls.
- `--debug-hand` — show a window with MediaPipe hand landmarks to help with
  troubleshooting.

Once running, flap your hand upward to make the bird hop through the pipes. The
score increases every time you pass a pair of pipes. When you crash, flap again
or press `R` to restart.

### Tips for reliable hand tracking

- Make sure your hand is fully inside the camera frame with good lighting.
- Keep your palm roughly facing the camera so the landmarks remain visible.
- Use quick upward flicks — the detector now looks for both total movement and
  the speed of your hand to catch shorter motions.
- If detection still seems off, run with `--debug-hand` to see the landmarks
  being tracked and adjust your positioning.
