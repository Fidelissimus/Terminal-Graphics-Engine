#!/usr/bin/env python3
"""
Terminal Graphics Engine

A high-performance, modular terminal graphics engine with advanced rendering
capabilities including anti-aliased graphics, true color support, and 3D rendering.

Features:
- Flicker-free double buffering with ANSI escape code optimization
- True color (24-bit) and 256-color support with intelligent color matching
- Cross-platform non-blocking input handling
- Comprehensive 2D/3D math library
- Advanced drawing primitives with anti-aliasing
- 3D wireframe rendering with perspective projection
"""

import os
import sys
import time
import math
import struct
import select
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Callable

# Platform-specific imports for input handling
try:
    import msvcrt  # Windows
except ImportError:
    import termios
    import tty     # Unix/Linux


@dataclass
class Vec2D:
    """2D vector for mathematical operations."""
    x: float
    y: float
    
    def __add__(self, other: 'Vec2D') -> 'Vec2D':
        """Vector addition."""
        return Vec2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vec2D') -> 'Vec2D':
        """Vector subtraction."""
        return Vec2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vec2D':
        """Scalar multiplication."""
        return Vec2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vec2D':
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def dot(self, other: 'Vec2D') -> float:
        """Dot product of two vectors."""
        return self.x * other.x + self.y * other.y
    
    def magnitude(self) -> float:
        """Vector magnitude (length)."""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> 'Vec2D':
        """Return normalized (unit) vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vec2D(0, 0)
        return Vec2D(self.x / mag, self.y / mag)
    
    def rotate(self, angle: float) -> 'Vec2D':
        """Rotate vector by angle (in radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)


@dataclass
class Vec3D:
    """3D vector for 3D mathematical operations."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vec3D') -> 'Vec3D':
        """Vector addition."""
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vec3D') -> 'Vec3D':
        """Vector subtraction."""
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vec3D':
        """Scalar multiplication."""
        return Vec3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vec3D':
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def dot(self, other: 'Vec3D') -> float:
        """Dot product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vec3D') -> 'Vec3D':
        """Cross product of two vectors."""
        return Vec3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        """Vector magnitude (length)."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self) -> 'Vec3D':
        """Return normalized (unit) vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vec3D(0, 0, 0)
        return Vec3D(self.x / mag, self.y / mag, self.z / mag)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)


class Matrix4x4:
    """
    4x4 matrix for 3D transformations and projections.
    
    The matrix is stored in row-major order:
    [m00, m01, m02, m03]
    [m10, m11, m12, m13]
    [m20, m21, m22, m23]
    [m30, m31, m32, m33]
    """
    
    def __init__(self, elements: Optional[List[float]] = None):
        """
        Initialize matrix.
        
        Args:
            elements: Optional 16-element list for matrix initialization.
                     If None, creates identity matrix.
        """
        if elements is None:
            self.m = [1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0]
        else:
            if len(elements) != 16:
                raise ValueError("Matrix4x4 requires exactly 16 elements")
            self.m = elements.copy()
    
    @classmethod
    def identity(cls) -> 'Matrix4x4':
        """Create identity matrix."""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create translation matrix."""
        return cls([1.0, 0.0, 0.0, x,
                    0.0, 1.0, 0.0, y,
                    0.0, 0.0, 1.0, z,
                    0.0, 0.0, 0.0, 1.0])
    
    @classmethod
    def rotation_x(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around X-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([1.0, 0.0,  0.0,   0.0,
                    0.0, cos_a, -sin_a, 0.0,
                    0.0, sin_a, cos_a,  0.0,
                    0.0, 0.0,  0.0,   1.0])
    
    @classmethod
    def rotation_y(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Y-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([cos_a,  0.0, sin_a, 0.0,
                    0.0,   1.0, 0.0,   0.0,
                    -sin_a, 0.0, cos_a, 0.0,
                    0.0,   0.0, 0.0,   1.0])
    
    @classmethod
    def rotation_z(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Z-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([cos_a, -sin_a, 0.0, 0.0,
                    sin_a, cos_a,  0.0, 0.0,
                    0.0,  0.0,   1.0, 0.0,
                    0.0,  0.0,   0.0, 1.0])
    
    @classmethod
    def perspective(cls, fov: float, aspect_ratio: float, near: float, far: float) -> 'Matrix4x4':
        """
        Create perspective projection matrix.
        
        Args:
            fov: Field of view in radians
            aspect_ratio: Width/height ratio
            near: Near clipping plane
            far: Far clipping plane
        """
        tan_half_fov = math.tan(fov / 2.0)
        range_inv = 1.0 / (near - far)
        
        return cls([1.0 / (aspect_ratio * tan_half_fov), 0.0, 0.0, 0.0,
                    0.0, 1.0 / tan_half_fov, 0.0, 0.0,
                    0.0, 0.0, (near + far) * range_inv, 2.0 * near * far * range_inv,
                    0.0, 0.0, -1.0, 0.0])
    
    def multiply_matrix(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Multiply this matrix by another matrix."""
        result = [0.0] * 16
        
        for i in range(4):
            for j in range(4):
                result[i * 4 + j] = (
                    self.m[i * 4 + 0] * other.m[0 * 4 + j] +
                    self.m[i * 4 + 1] * other.m[1 * 4 + j] +
                    self.m[i * 4 + 2] * other.m[2 * 4 + j] +
                    self.m[i * 4 + 3] * other.m[3 * 4 + j]
                )
        
        return Matrix4x4(result)
    
    def multiply_vector(self, v: Vec3D) -> Vec3D:
        """Multiply this matrix by a 3D vector (treating it as position)."""
        x = self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z + self.m[3]
        y = self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z + self.m[7]
        z = self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z + self.m[11]
        w = self.m[12] * v.x + self.m[13] * v.y + self.m[14] * v.z + self.m[15]
        
        if w != 0.0:
            return Vec3D(x / w, y / w, z / w)
        return Vec3D(x, y, z)


@dataclass
class Cell:
    """
    Represents a single character cell in the terminal buffer.
    
    Attributes:
        char: Character to display
        fg: Foreground color as (R, G, B) tuple
        bg: Background color as (R, G, B) tuple
    """
    char: str = ' '
    fg: Tuple[int, int, int] = (255, 255, 255)  # White
    bg: Tuple[int, int, int] = (0, 0, 0)        # Black


class Sprite:
    """
    2D sprite with transparency support.
    
    A sprite is a 2D array of Cell objects that can be drawn onto the buffer.
    None values in the data array represent transparent pixels.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize sprite.
        
        Args:
            width: Sprite width in characters
            height: Sprite height in characters
        """
        self.width = width
        self.height = height
        self.data = [[None for _ in range(width)] for _ in range(height)]
    
    def set_pixel(self, x: int, y: int, char: str, fg: Tuple[int, int, int], bg: Tuple[int, int, int]):
        """
        Set sprite pixel.
        
        Args:
            x: X coordinate
            y: Y coordinate
            char: Character to display
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y][x] = Cell(char, fg, bg)
    
    def clear_pixel(self, x: int, y: int):
        """
        Clear sprite pixel (make transparent).
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y][x] = None


class TerminalInput:
    """
    Cross-platform non-blocking input handler.
    
    Provides unified interface for keyboard input across Windows and Unix systems.
    """
    
    def __init__(self):
        """Initialize input handler."""
        self._original_terminal_state = None
        self._is_windows = os.name == 'nt'
    
    def setup_terminal(self):
        """Configure terminal for non-blocking input."""
        if not self._is_windows:
            # Save original terminal state
            self._original_terminal_state = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
    
    def restore_terminal(self):
        """Restore original terminal state."""
        if not self._is_windows and self._original_terminal_state:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original_terminal_state)
    
    @contextlib.contextmanager
    def terminal_context(self):
        """
        Context manager for terminal input handling.
        
        Ensures terminal is properly restored even if exceptions occur.
        """
        try:
            self.setup_terminal()
            yield self
        finally:
            self.restore_terminal()
    
    def get_key(self) -> Optional[str]:
        """
        Get pressed key without blocking.
        
        Returns:
            String representing pressed key, or None if no key pressed.
            Special keys are represented as strings like 'esc', 'up', 'down', etc.
        """
        if self._is_windows:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Handle special keys on Windows
                if key == b'\xe0':  # Extended key
                    key = msvcrt.getch()
                    return self._decode_windows_extended_key(key)
                return key.decode('utf-8', errors='ignore')
        else:
            # Unix/Linux
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                if key == '\x1b':  # Escape sequence
                    # Check for extended escape sequences
                    if select.select([sys.stdin], [], [], 0.01) == ([sys.stdin], [], []):
                        next_chars = sys.stdin.read(2)
                        return self._decode_unix_escape_sequence(next_chars)
                    return 'esc'
                return key
        return None
    
    def _decode_windows_extended_key(self, key: bytes) -> str:
        """Decode Windows extended key codes."""
        key_map = {
            b'H': 'up',
            b'P': 'down',
            b'K': 'left',
            b'M': 'right',
            b'G': 'home',
            b'O': 'end',
            b'R': 'insert',
            b'S': 'delete',
            b'I': 'pageup',
            b'Q': 'pagedown'
        }
        return key_map.get(key, 'unknown')
    
    def _decode_unix_escape_sequence(self, sequence: str) -> str:
        """Decode Unix escape sequences."""
        key_map = {
            '[A': 'up',
            '[B': 'down',
            '[C': 'right',
            '[D': 'left',
            '[H': 'home',
            '[F': 'end',
            '[2~': 'insert',
            '[3~': 'delete',
            '[5~': 'pageup',
            '[6~': 'pagedown'
        }
        return key_map.get(sequence, 'unknown')


def rgb_to_xterm256(r: int, g: int, b: int) -> int:
    """
    Convert RGB color to closest xterm 256-color index.
    
    This function intelligently maps 24-bit RGB colors to the closest
    available color in the xterm 256-color palette.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        
    Returns:
        xterm 256-color index (0-255)
    """
    # Handle grayscale first (more precise matching)
    if r == g == b:
        # Map to grayscale ramp (232-255)
        gray_value = (r + g + b) / 3
        gray_index = int((gray_value / 255) * 23) + 232
        return min(max(gray_index, 232), 255)
    
    # Map to 6x6x6 color cube (16-231)
    r_index = int(r / 255 * 5)
    g_index = int(g / 255 * 5)
    b_index = int(b / 255 * 5)
    
    # Clamp indices
    r_index = max(0, min(5, r_index))
    g_index = max(0, min(5, g_index))
    b_index = max(0, min(5, b_index))
    
    return 16 + 36 * r_index + 6 * g_index + b_index


class TerminalGraphicsEngine:
    """
    High-performance terminal graphics engine.
    
    Features double buffering, true color support, advanced drawing primitives,
    and cross-platform input handling.
    """
    
    def __init__(self, width: int = 120, height: int = 40, use_true_color: bool = True):
        """
        Initialize graphics engine.
        
        Args:
            width: Terminal width in characters
            height: Terminal height in characters
            use_true_color: Whether to use 24-bit true color (if supported)
        """
        self.width = width
        self.height = height
        self.use_true_color = use_true_color
        
        # Double buffering system
        self.front_buffer = [[Cell() for _ in range(width)] for _ in range(height)]
        self.back_buffer = [[Cell() for _ in range(width)] for _ in range(height)]
        
        # Input handler
        self.input_handler = TerminalInput()
        
        # Performance monitoring
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # ANSI escape codes for optimized rendering
        self._ansi_hide_cursor = '\033[?25l'
        self._ansi_show_cursor = '\033[?25h'
        self._ansi_home = '\033[H'
        self._ansi_clear = '\033[2J'
        
        # Character gradient for intensity mapping
        self.chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    def __enter__(self):
        """Context manager entry."""
        self.input_handler.setup_terminal()
        # Hide cursor
        print(self._ansi_hide_cursor, end='', flush=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Show cursor and restore terminal
        print(self._ansi_show_cursor, end='', flush=True)
        self.input_handler.restore_terminal()
    
    def clear(self):
        """Clear back buffer."""
        self.back_buffer = [[Cell() for _ in range(self.width)] for _ in range(self.height)]
    
    def set_pixel(self, x: int, y: int, char: str = '█', 
                  fg: Tuple[int, int, int] = (255, 255, 255),
                  bg: Tuple[int, int, int] = (0, 0, 0)):
        """
        Set pixel in back buffer.
        
        Args:
            x: X coordinate
            y: Y coordinate
            char: Character to display
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.back_buffer[y][x] = Cell(char, fg, bg)
    
    def draw_line_fast(self, x1: int, y1: int, x2: int, y2: int,
                       char: str = '█', fg: Tuple[int, int, int] = (255, 255, 255),
                       bg: Tuple[int, int, int] = (0, 0, 0)):
        """
        Draw line using Bresenham's algorithm (fast, aliased).
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            char: Character to use for line
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            self.set_pixel(x1, y1, char, fg, bg)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def draw_line_antialiased(self, x1: float, y1: float, x2: float, y2: float,
                              char: str = '█', fg: Tuple[int, int, int] = (255, 255, 255),
                              bg: Tuple[int, int, int] = (0, 0, 0)):
        """
        Draw anti-aliased line using Xiaolin Wu's algorithm.
        
        Args:
            x1, y1: Start coordinates (floats for sub-pixel positioning)
            x2, y2: End coordinates (floats for sub-pixel positioning)
            char: Character to use for line
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        # Implementation of Xiaolin Wu's line algorithm
        steep = abs(y2 - y1) > abs(x2 - x1)
        
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            gradient = 1.0
        else:
            gradient = dy / dx
        
        # Handle first endpoint
        xend = round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = 1 - (x1 + 0.5) % 1
        xpxl1 = xend
        ypxl1 = int(yend)
        
        if steep:
            self._plot_aa(ypxl1, xpxl1, 1 - (yend % 1) * xgap, char, fg, bg)
            self._plot_aa(ypxl1 + 1, xpxl1, (yend % 1) * xgap, char, fg, bg)
        else:
            self._plot_aa(xpxl1, ypxl1, 1 - (yend % 1) * xgap, char, fg, bg)
            self._plot_aa(xpxl1, ypxl1 + 1, (yend % 1) * xgap, char, fg, bg)
        
        intery = yend + gradient
        
        # Handle second endpoint
        xend = round(x2)
        yend = y2 + gradient * (xend - x2)
        xgap = (x2 + 0.5) % 1
        xpxl2 = xend
        ypxl2 = int(yend)
        
        if steep:
            self._plot_aa(ypxl2, xpxl2, 1 - (yend % 1) * xgap, char, fg, bg)
            self._plot_aa(ypxl2 + 1, xpxl2, (yend % 1) * xgap, char, fg, bg)
        else:
            self._plot_aa(xpxl2, ypxl2, 1 - (yend % 1) * xgap, char, fg, bg)
            self._plot_aa(xpxl2, ypxl2 + 1, (yend % 1) * xgap, char, fg, bg)
        
        # Main loop
        for x in range(int(xpxl1) + 1, int(xpxl2)):
            if steep:
                self._plot_aa(int(intery), x, 1 - (intery % 1), char, fg, bg)
                self._plot_aa(int(intery) + 1, x, intery % 1, char, fg, bg)
            else:
                self._plot_aa(x, int(intery), 1 - (intery % 1), char, fg, bg)
                self._plot_aa(x, int(intery) + 1, intery % 1, char, fg, bg)
            intery += gradient
    
    def _plot_aa(self, x: int, y: int, brightness: float, char: str,
                 fg: Tuple[int, int, int], bg: Tuple[int, int, int]):
        """
        Plot anti-aliased pixel with brightness adjustment.
        
        Args:
            x, y: Coordinates
            brightness: Pixel brightness (0.0-1.0)
            char: Character to display
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            # Blend foreground with background based on brightness
            blended_fg = (
                int(fg[0] * brightness + bg[0] * (1 - brightness)),
                int(fg[1] * brightness + bg[1] * (1 - brightness)),
                int(fg[2] * brightness + bg[2] * (1 - brightness))
            )
            self.set_pixel(x, y, char, blended_fg, bg)
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int,
                       char: str = '█', fg: Tuple[int, int, int] = (255, 255, 255),
                       bg: Tuple[int, int, int] = (0, 0, 0), filled: bool = False):
        """
        Draw rectangle.
        
        Args:
            x, y: Top-left corner
            width: Rectangle width
            height: Rectangle height
            char: Character to use
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
            filled: Whether to fill the rectangle
        """
        if filled:
            for py in range(y, y + height):
                for px in range(x, x + width):
                    self.set_pixel(px, py, char, fg, bg)
        else:
            # Draw four sides
            for px in range(x, x + width):
                self.set_pixel(px, y, char, fg, bg)
                self.set_pixel(px, y + height - 1, char, fg, bg)
            for py in range(y, y + height):
                self.set_pixel(x, py, char, fg, bg)
                self.set_pixel(x + width - 1, py, char, fg, bg)
    
    def draw_circle(self, cx: int, cy: int, radius: int,
                    char: str = '█', fg: Tuple[int, int, int] = (255, 255, 255),
                    bg: Tuple[int, int, int] = (0, 0, 0), filled: bool = False):
        """
        Draw circle using midpoint algorithm.
        
        Args:
            cx, cy: Center coordinates
            radius: Circle radius
            char: Character to use
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
            filled: Whether to fill the circle
        """
        x = radius
        y = 0
        err = 0
        
        while x >= y:
            # Draw all eight octants
            points = [
                (cx + x, cy + y), (cx + y, cy + x), (cx - y, cy + x), (cx - x, cy + y),
                (cx - x, cy - y), (cx - y, cy - x), (cx + y, cy - x), (cx + x, cy - y)
            ]
            
            for px, py in points:
                self.set_pixel(px, py, char, fg, bg)
            
            if filled:
                # Fill horizontal lines for filled circle
                for fill_x in range(cx - x, cx + x + 1):
                    self.set_pixel(fill_x, cy + y, char, fg, bg)
                    self.set_pixel(fill_x, cy - y, char, fg, bg)
                for fill_x in range(cx - y, cx + y + 1):
                    self.set_pixel(fill_x, cy + x, char, fg, bg)
                    self.set_pixel(fill_x, cy - x, char, fg, bg)
            
            y += 1
            err += 1 + 2 * y
            if 2 * (err - x) + 1 > 0:
                x -= 1
                err += 1 - 2 * x
    
    def draw_filled_polygon(self, points: List[Vec2D],
                            char: str = '█', fg: Tuple[int, int, int] = (255, 255, 255),
                            bg: Tuple[int, int, int] = (0, 0, 0)):
        """
        Draw filled polygon using scanline algorithm.
        
        Args:
            points: List of vertices defining the polygon
            char: Character to use
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        if len(points) < 3:
            return  # Need at least 3 points for a polygon
        
        # Find polygon bounds
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        # Process each scanline
        for y in range(int(min_y), int(max_y) + 1):
            intersections = []
            
            # Find intersections with each edge
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                # Skip horizontal edges
                if p1.y == p2.y:
                    continue
                
                # Ensure p1 is above p2
                if p1.y > p2.y:
                    p1, p2 = p2, p1
                
                # Check if scanline intersects this edge
                if p1.y <= y < p2.y:
                    # Calculate intersection x coordinate
                    t = (y - p1.y) / (p2.y - p1.y)
                    x_intersect = p1.x + t * (p2.x - p1.x)
                    intersections.append(x_intersect)
            
            # Sort intersections and draw between pairs
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    start_x = int(intersections[i])
                    end_x = int(intersections[i + 1])
                    for x in range(start_x, end_x + 1):
                        self.set_pixel(x, y, char, fg, bg)
    
    def draw_text(self, x: int, y: int, text: str,
                  fg: Tuple[int, int, int] = (255, 255, 255),
                  bg: Tuple[int, int, int] = (0, 0, 0)):
        """
        Draw text at specified position.
        
        Args:
            x, y: Starting position
            text: Text to draw
            fg: Foreground color (R, G, B)
            bg: Background color (R, G, B)
        """
        for i, char in enumerate(text):
            self.set_pixel(x + i, y, char, fg, bg)
    
    def draw_sprite(self, x: int, y: int, sprite: Sprite):
        """
        Draw sprite at specified position.
        
        Args:
            x, y: Top-left position
            sprite: Sprite to draw
        """
        for sprite_y in range(sprite.height):
            for sprite_x in range(sprite.width):
                cell = sprite.data[sprite_y][sprite_x]
                if cell is not None:  # Skip transparent pixels
                    screen_x = x + sprite_x
                    screen_y = y + sprite_y
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        self.back_buffer[screen_y][screen_x] = cell
    
    def project_3d_to_2d(self, point: Vec3D, 
                         model_matrix: Matrix4x4,
                         view_matrix: Matrix4x4,
                         projection_matrix: Matrix4x4) -> Vec2D:
        """
        Project 3D point to 2D screen coordinates.
        
        Args:
            point: 3D point to project
            model_matrix: Model transformation matrix
            view_matrix: View (camera) matrix
            projection_matrix: Projection matrix
            
        Returns:
            2D screen coordinates
        """
        # Apply transformations: model -> view -> projection
        transformed = model_matrix.multiply_vector(point)
        transformed = view_matrix.multiply_vector(transformed)
        transformed = projection_matrix.multiply_vector(transformed)
        
        # Convert to screen coordinates
        screen_x = int((transformed.x + 1) * 0.5 * self.width)
        screen_y = int((1 - (transformed.y + 1) * 0.5) * self.height)
        
        return Vec2D(screen_x, screen_y)
    
    def _calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def _compile_frame_string(self) -> str:
        """
        Compile back buffer into ANSI-escaped frame string.
        
        Returns:
            ANSI-escaped string representing the complete frame
        """
        frame_lines = []
        current_fg = None
        current_bg = None
        
        for y, row in enumerate(self.back_buffer):
            line_parts = []
            
            for x, cell in enumerate(row):
                # Only emit color codes when colors change (optimization)
                fg_code = self._get_color_code(cell.fg, is_background=False)
                bg_code = self._get_color_code(cell.bg, is_background=True)
                
                if fg_code != current_fg or bg_code != current_bg:
                    # Reset colors and set new ones
                    if current_fg is not None or current_bg is not None:
                        line_parts.append('\033[0m')  # Reset
                    
                    line_parts.append(fg_code)
                    line_parts.append(bg_code)
                    current_fg = fg_code
                    current_bg = bg_code
                
                line_parts.append(cell.char)
            
            # Reset at end of line
            if current_fg is not None or current_bg is not None:
                line_parts.append('\033[0m')
                current_fg = None
                current_bg = None
            
            frame_lines.append(''.join(line_parts))
        
        return '\n'.join(frame_lines)
    
    def _get_color_code(self, rgb: Tuple[int, int, int], is_background: bool = False) -> str:
        """
        Get ANSI color code for RGB color.
        
        Args:
            rgb: (R, G, B) tuple
            is_background: Whether this is a background color
            
        Returns:
            ANSI escape code for the color
        """
        r, g, b = rgb
        
        if self.use_true_color:
            # 24-bit true color
            color_type = 48 if is_background else 38
            return f'\033[{color_type};2;{r};{g};{b}m'
        else:
            # 256-color mode
            color_index = rgb_to_xterm256(r, g, b)
            color_type = 48 if is_background else 38
            return f'\033[{color_type};5;{color_index}m'
    
    def render(self):
        """Render back buffer to terminal with double buffering."""
        # Calculate FPS
        self._calculate_fps()
        
        # Compile frame string
        frame_string = self._compile_frame_string()
        
        # Use ANSI escape codes for flicker-free rendering
        render_output = self._ansi_home + frame_string
        
        # Print FPS counter
        fps_text = f" FPS: {self.fps:.1f} "
        render_output += f'\033[1;1H\033[47m\033[30m{fps_text}\033[0m'
        
        # Output frame in one operation
        print(render_output, end='', flush=True)
        
        # Swap buffers
        self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
        
        # Clear back buffer for next frame
        self.clear()


def main():
    """Demo function showcasing engine capabilities."""
    # Create engine instance
    with TerminalGraphicsEngine(width=120, height=40, use_true_color=True) as engine:
        # Demo variables
        angle = 0.0
        last_time = time.time()
        
        print("Terminal Graphics Engine Demo")
        print("Rendering: Anti-aliased lines, filled polygon, spinning 3D cube")
        print("Press 'q' to exit, 't' to toggle true color mode")
        
        # Main demo loop
        while True:
            # Handle input
            key = engine.input_handler.get_key()
            if key == 'q':
                break
            elif key == 't':
                engine.use_true_color = not engine.use_true_color
            
            # Clear back buffer
            engine.clear()
            
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            angle += delta_time * 0.5  # Rotate 0.5 radians per second
            
            # Demo 1: Anti-aliased lines (rainbow effect)
            center_x, center_y = engine.width // 4, engine.height // 2
            radius = min(center_x, center_y) - 2
            
            for i in range(36):
                angle1 = i * math.pi / 18
                angle2 = (i + 10) * math.pi / 18
                
                # Rainbow colors
                r = int(127 + 127 * math.sin(angle1))
                g = int(127 + 127 * math.sin(angle1 + 2 * math.pi / 3))
                b = int(127 + 127 * math.sin(angle1 + 4 * math.pi / 3))
                
                x1 = center_x + radius * math.cos(angle1)
                y1 = center_y + radius * math.sin(angle1)
                x2 = center_x + radius * math.cos(angle2)
                y2 = center_y + radius * math.sin(angle2)
                
                engine.draw_line_antialiased(
                    x1, y1, x2, y2,
                    '·', (r, g, b), (0, 0, 0)
                )
            
            # Demo 2: Filled polygon (rotating hexagon)
            hex_center_x = engine.width * 3 // 4
            hex_center_y = engine.height // 4
            hex_radius = 8
            
            hex_points = []
            for i in range(6):
                hex_angle = angle + i * math.pi / 3
                hex_points.append(Vec2D(
                    hex_center_x + hex_radius * math.cos(hex_angle),
                    hex_center_y + hex_radius * math.sin(hex_angle)
                ))
            
            # Gradient color based on rotation
            poly_color = (
                int(127 + 127 * math.sin(angle)),
                int(127 + 127 * math.sin(angle + math.pi / 2)),
                int(127 + 127 * math.sin(angle + math.pi))
            )
            
            engine.draw_filled_polygon(hex_points, '▒', poly_color, (20, 20, 40))
            
            # Demo 3: 3D wireframe cube
            cube_center_x = engine.width * 3 // 4
            cube_center_y = engine.height * 3 // 4
            cube_size = 5
            
            # Define cube vertices
            cube_vertices = [
                Vec3D(-1, -1, -1), Vec3D(1, -1, -1), Vec3D(1, 1, -1), Vec3D(-1, 1, -1),
                Vec3D(-1, -1, 1), Vec3D(1, -1, 1), Vec3D(1, 1, 1), Vec3D(-1, 1, 1)
            ]
            
            # Scale vertices
            cube_vertices = [v * cube_size for v in cube_vertices]
            
            # Define cube edges
            cube_edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
            ]
            
            # Create transformation matrices
            model_matrix = (Matrix4x4.rotation_y(angle)
                          .multiply_matrix(Matrix4x4.rotation_x(angle * 0.7)))
            
            # Move cube away from camera
            model_matrix = (Matrix4x4.translation(0, 0, 10)
                          .multiply_matrix(model_matrix))
            
            # Simple view matrix (identity for now)
            view_matrix = Matrix4x4.identity()
            
            # Perspective projection
            aspect_ratio = engine.width / engine.height
            projection_matrix = Matrix4x4.perspective(
                math.pi / 3, aspect_ratio, 0.1, 100.0
            )
            
            # Project vertices to 2D
            projected_vertices = []
            for vertex in cube_vertices:
                screen_pos = engine.project_3d_to_2d(
                    vertex, model_matrix, view_matrix, projection_matrix
                )
                # Offset to cube center.. not needed
                ##screen_pos.x += cube_center_x
                ##screen_pos.y += cube_center_y
                projected_vertices.append(screen_pos)
            
            # Draw cube edges
            cube_color = (200, 150, 100)
            for edge in cube_edges:
                v1 = projected_vertices[edge[0]]
                v2 = projected_vertices[edge[1]]
                engine.draw_line_fast(
                    int(v1.x), int(v1.y), int(v2.x), int(v2.y),
                    '≣', cube_color, (0, 0, 0)
                )
            
            # Draw info text
            color_mode = "True Color" if engine.use_true_color else "256-Color"
            engine.draw_text(2, engine.height - 3, f"Color Mode: {color_mode}", (255, 255, 255), (0, 0, 0))
            engine.draw_text(2, engine.height - 2, "Press 'q' to quit, 't' to toggle color mode", (200, 200, 200), (0, 0, 0))
            
            # Render frame
            engine.render()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)


if __name__ == "__main__":

    main()
