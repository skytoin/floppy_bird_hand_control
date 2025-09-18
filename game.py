"""Pygame implementation of a Floppy Bird style game controlled by hand flaps."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import pygame

from hand_control import HandFlapDetector

# Screen configuration
WIDTH = 480
HEIGHT = 720
FPS = 60

# Bird physics
GRAVITY = 1100.0  # Pixels / second^2
FLAP_VELOCITY = -360.0
MAX_DESCENT_SPEED = 420.0

# Pipes
PIPE_SPEED = 170.0
PIPE_SPAWN_INTERVAL = 1800  # milliseconds
PIPE_GAP = 220
PIPE_WIDTH = 70

BACKGROUND_COLOR = (15, 20, 40)
BIRD_COLOR = (255, 219, 88)
PIPE_COLOR = (52, 168, 83)
TEXT_COLOR = (230, 230, 230)
OVERLAY_COLOR = (0, 0, 0, 160)


@dataclass
class Bird:
    x: float
    y: float
    radius: int = 22
    velocity: float = 0.0

    def flap(self) -> None:
        self.velocity = FLAP_VELOCITY

    def update(self, dt: float) -> None:
        self.velocity = min(self.velocity + GRAVITY * dt, MAX_DESCENT_SPEED)
        self.y += self.velocity * dt

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False
    width: int = PIPE_WIDTH
    gap_size: int = PIPE_GAP

    @property
    def top_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), 0, self.width, int(self.gap_y - self.gap_size / 2))

    @property
    def bottom_rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x),
            int(self.gap_y + self.gap_size / 2),
            self.width,
            HEIGHT - int(self.gap_y + self.gap_size / 2),
        )

    def update(self, dt: float) -> None:
        self.x -= PIPE_SPEED * dt

    def is_off_screen(self) -> bool:
        return self.x + self.width < 0


class FloppyBirdGame:
    def __init__(self, *, enable_hand_control: bool = True, debug_hand: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("Floppy Bird - Hand Controlled")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("arial", 48, bold=True)
        self.font_medium = pygame.font.SysFont("arial", 32)
        self.font_small = pygame.font.SysFont("arial", 22)

        self.detector: Optional[HandFlapDetector] = None
        if enable_hand_control:
            try:
                self.detector = HandFlapDetector(debug=debug_hand)
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"[WARN] Hand control disabled: {exc}")
                self.detector = None


        self.reset()
        self._spawn_event = pygame.USEREVENT + 1
        pygame.time.set_timer(self._spawn_event, PIPE_SPAWN_INTERVAL)

    def reset(self) -> None:
        self.bird = Bird(x=WIDTH * 0.25, y=HEIGHT / 2)
        self.pipes: List[Pipe] = []
        self.score = 0
        self.game_over = False
        self._time_since_over = 0.0

    def spawn_pipe(self) -> None:
        margin = 120
        gap_y = random.uniform(margin, HEIGHT - margin)
        self.pipes.append(Pipe(x=WIDTH + PIPE_WIDTH, gap_y=gap_y))

    def update_game(self, dt: float) -> None:
        if self.game_over:
            self._time_since_over += dt
            return

        self.bird.update(dt)

        for pipe in self.pipes:
            pipe.update(dt)

        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]

        # Collision detection
        if self._collides_with_pipes(self.bird):
            self.game_over = True
            self._time_since_over = 0.0

        if self.bird.y - self.bird.radius < 0 or self.bird.y + self.bird.radius > HEIGHT:
            self.game_over = True
            self._time_since_over = 0.0

        for pipe in self.pipes:
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1

    def _collides_with_pipes(self, bird: Bird) -> bool:
        bird_rect = bird.rect
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe.top_rect) or bird_rect.colliderect(pipe.bottom_rect):
                return True
        return False

    def process_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._shutdown()
                    raise SystemExit
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if self.game_over:
                        self.reset()
                    self.bird.flap()
                if event.key == pygame.K_r:
                    self.reset()
            if event.type == self._spawn_event and not self.game_over:
                self.spawn_pipe()

        # Poll the hand detector outside of the event loop so that multiple
        # flaps can be registered per frame.
        if self.detector and self.detector.poll_flap():
            if self.game_over:
                self.reset()
            self.bird.flap()

    def draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        for pipe in self.pipes:
            pygame.draw.rect(self.screen, PIPE_COLOR, pipe.top_rect)
            pygame.draw.rect(self.screen, PIPE_COLOR, pipe.bottom_rect)

        pygame.draw.circle(self.screen, BIRD_COLOR, (int(self.bird.x), int(self.bird.y)), self.bird.radius)

        score_surface = self.font_large.render(str(self.score), True, TEXT_COLOR)
        self.screen.blit(score_surface, score_surface.get_rect(center=(WIDTH / 2, 80)))

        if self.detector is None:
            info = self.font_small.render("Press SPACE/UP to flap", True, TEXT_COLOR)
            self.screen.blit(info, info.get_rect(center=(WIDTH / 2, HEIGHT - 40)))
        else:
            info_lines = [
                "Flap by raising your hand!",
                "Press SPACE if the camera fails to pick up.",
            ]
            for i, line in enumerate(info_lines):
                info_surface = self.font_small.render(line, True, TEXT_COLOR)
                self.screen.blit(info_surface, info_surface.get_rect(center=(WIDTH / 2, HEIGHT - 40 + i * 24)))

        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill(OVERLAY_COLOR)
            self.screen.blit(overlay, (0, 0))
            game_over_surface = self.font_large.render("Game Over", True, TEXT_COLOR)
            retry_surface = self.font_medium.render("Flap or press R to retry", True, TEXT_COLOR)
            self.screen.blit(game_over_surface, game_over_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 20)))
            self.screen.blit(retry_surface, retry_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 40)))

        pygame.display.flip()

    def run(self) -> None:
        try:
            while True:
                dt = self.clock.tick(FPS) / 1000.0
                self.process_input()
                self.update_game(dt)
                self.draw()
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        if self.detector is not None:
            self.detector.stop()
        pygame.quit()


__all__ = ["FloppyBirdGame"]
