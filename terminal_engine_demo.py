#!/usr/bin/env python3
"""
Terminal Graphics Engine Demo Suite

A comprehensive showcase of the terminal graphics engine capabilities,
featuring multiple interactive demos that highlight different aspects
of the engine's functionality.
"""

import math
import time
from terminal_engine import *


def run_2d_primitives_demo(engine):
    """
    Demo 1: 2D Primitives & Anti-Aliasing Showcase
    
    Demonstrates the quality difference between fast (aliased) and 
    anti-aliased drawing primitives, along with filled polygons.
    """
    last_time = time.time()
    rotation = 0.0
    
    print("2D Primitives & Anti-Aliasing Demo")
    print("Left: Fast (aliased) | Right: Anti-aliased")
    print("Press 'q' to return to menu")
    
    while True:
        # Handle input
        key = engine.input_handler.get_key()
        if key == 'q':
            break
        
        # Calculate delta time and rotation
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        rotation += delta_time * 0.5
        
        # Clear back buffer
        engine.clear()
        
        # Calculate screen division
        mid_x = engine.width // 2
        mid_y = engine.height // 2
        
        # Draw dividing line
        for y in range(engine.height):
            engine.set_pixel(mid_x, y, '│', (100, 100, 100))
        
        # Draw labels
        engine.draw_text(mid_x // 4 - 8, 2, "Bresenham (Fast)", (255, 200, 100))
        engine.draw_text(mid_x + mid_x // 4 - 8, 2, "Xiaolin Wu (Smooth)", (100, 200, 255))
        
        # Demo parameters
        center_left = Vec2D(mid_x // 2, mid_y)
        center_right = Vec2D(mid_x + mid_x // 2, mid_y)
        size = min(mid_x, mid_y) * 0.8
        
        # Demo 1: Rotating lines
        for i in range(8):
            angle = rotation + i * math.pi / 4
            color = (
                int(127 + 127 * math.sin(angle)),
                int(127 + 127 * math.sin(angle + math.pi / 3)),
                int(127 + 127 * math.sin(angle + 2 * math.pi / 3))
            )
            
            # Fast line (left side)
            x1_fast = center_left.x + size * 0.8 * math.cos(angle)
            y1_fast = center_left.y + size * 0.8 * math.sin(angle)
            engine.draw_line_fast(
                int(center_left.x), int(center_left.y), 
                int(x1_fast), int(y1_fast),
                '█', color, (0, 0, 0)
            )
            
            # Anti-aliased line (right side)
            x1_aa = center_right.x + size * 0.8 * math.cos(angle)
            y1_aa = center_right.y + size * 0.8 * math.sin(angle)
            engine.draw_line_antialiased(
                center_right.x, center_right.y, 
                x1_aa, y1_aa,
                '█', color, (0, 0, 0)
            )
        
        # Demo 2: Circles
        circle_radius = size * 0.3
        
        # Fast circle (left side)
        engine.draw_circle(
            int(center_left.x), int(center_left.y), 
            int(circle_radius),
            '·', (255, 100, 100), (0, 0, 0), filled=False
        )
        
        # Anti-aliased circle effect (right side) - we'll draw multiple circles with different chars
        for r in range(int(circle_radius) - 2, int(circle_radius) + 3):
            intensity = 1.0 - abs(r - circle_radius) / 3.0
            if intensity > 0:
                color = (100, 255, 100)
                aa_color = (
                    int(color[0] * intensity),
                    int(color[1] * intensity),
                    int(color[2] * intensity)
                )
                chars = "·•"
                char = chars[int(intensity * len(chars)) % len(chars)]
                
                # Use multiple draw_circle calls with different radii to simulate AA
                engine.draw_circle(
                    int(center_right.x), int(center_right.y), 
                    r,
                    char, aa_color, (0, 0, 0), filled=False
                )
        
        # Demo 3: Rectangles
        rect_size = size * 0.4
        
        # Fast rectangle (left side)
        engine.draw_rectangle(
            int(center_left.x - rect_size), int(center_left.y - rect_size),
            int(rect_size * 2), int(rect_size * 2),
            '█', (100, 100, 255), (0, 0, 0), filled=False
        )
        
        # Anti-aliased rectangle effect (right side)
        for i in range(3):
            offset = i * 0.7
            intensity = 1.0 - i * 0.3
            color = (100, 100, 255)
            aa_color = (
                int(color[0] * intensity),
                int(color[1] * intensity),
                int(color[2] * intensity)
            )
            
            engine.draw_rectangle(
                int(center_right.x - rect_size + offset), 
                int(center_right.y - rect_size + offset),
                int(rect_size * 2 - offset * 2), 
                int(rect_size * 2 - offset * 2),
                '▒', aa_color, (0, 0, 0), filled=False
            )
        
        # Demo 4: Filled polygon (complex shape)
        poly_center = Vec2D(mid_x, engine.height * 3 // 4)
        poly_radius = size * 0.7
        num_points = 7
        
        poly_points = []
        for i in range(num_points):
            angle = rotation + i * 2 * math.pi / num_points
            # Vary radius for star-like shape
            point_radius = poly_radius if i % 2 == 0 else poly_radius * 0.5
            poly_points.append(Vec2D(
                poly_center.x + point_radius * math.cos(angle),
                poly_center.y + point_radius * math.sin(angle)
            ))
        
        # Multi-colored gradient for polygon
        poly_color = (
            int(127 + 127 * math.sin(rotation * 2)),
            int(127 + 127 * math.sin(rotation * 2 + math.pi / 2)),
            int(127 + 127 * math.sin(rotation * 2 + math.pi))
        )
        
        engine.draw_filled_polygon(
            poly_points, '▓', poly_color, (30, 20, 40)
        )
        
        # Draw polygon outline
        for i in range(num_points):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % num_points]
            engine.draw_line_antialiased(
                p1.x, p1.y, p2.x, p2.y,
                '░', (255, 255, 255), (30, 20, 40)
            )
        
        # Render frame
        engine.render()
        
        # Small delay
        time.sleep(0.02)


def run_3d_projection_demo(engine):
    """
    Demo 2: 3D Math & Projection Showcase
    
    Demonstrates the 3D math library with a complex wireframe object
    (icosahedron) rendered with smooth anti-aliased lines.
    """
    last_time = time.time()
    rotation_x = 0.0
    rotation_y = 0.0
    
    print("3D Math & Projection Demo")
    print("Rendering: Rotating Icosahedron with Anti-Aliased Edges")
    print("Use 'w', 's', 'a', 'd' to rotate | 'q' to return to menu")
    
    # Create icosahedron vertices (12 vertices of a regular icosahedron)
    t = (1.0 + math.sqrt(5.0)) / 2.0
    icosa_vertices = [
        Vec3D(-1,  t,  0), Vec3D( 1,  t,  0), Vec3D(-1, -t,  0), Vec3D( 1, -t,  0),
        Vec3D( 0, -1,  t), Vec3D( 0,  1,  t), Vec3D( 0, -1, -t), Vec3D( 0,  1, -t),
        Vec3D( t,  0, -1), Vec3D( t,  0,  1), Vec3D(-t,  0, -1), Vec3D(-t,  0,  1)
    ]
    
    # Normalize vertices to unit sphere
    icosa_vertices = [v.normalize() for v in icosa_vertices]
    
    # Define icosahedron edges (30 edges)
    icosa_edges = [
        (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 9), (1, 7),
        (2, 3), (2, 4), (2, 6), (2, 10), (2, 11),
        (3, 4), (3, 6), (3, 8), (3, 9),
        (4, 5), (4, 9), (4, 11),
        (5, 9), (5, 11),
        (6, 7), (6, 8), (6, 10),
        (7, 8), (7, 10),
        (8, 9),
        (10, 11)
    ]
    
    while True:
        # Handle input
        key = engine.input_handler.get_key()
        if key == 'q':
            break
        elif key == 'w':
            rotation_x += 0.1
        elif key == 's':
            rotation_x -= 0.1
        elif key == 'a':
            rotation_y += 0.1
        elif key == 'd':
            rotation_y -= 0.1
        
        # Calculate delta time
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        # Auto-rotation
        rotation_x += delta_time * 0.3
        rotation_y += delta_time * 0.2
        
        # Clear back buffer
        engine.clear()
        
        # Create transformation matrices
        model_matrix = (Matrix4x4.rotation_y(rotation_y)
                      .multiply_matrix(Matrix4x4.rotation_x(rotation_x)))
        
        # Scale and position the object
        model_matrix = (Matrix4x4.translation(0, 0, 8)
                      .multiply_matrix(model_matrix))
        
        # View matrix (identity for simplicity)
        view_matrix = Matrix4x4.identity()
        
        # Perspective projection
        aspect_ratio = engine.width / engine.height
        projection_matrix = Matrix4x4.perspective(
            math.pi / 3, aspect_ratio, 0.1, 100.0
        )
        
        # Project vertices to 2D
        projected_vertices = []
        for vertex in icosa_vertices:
            # Scale the icosahedron
            scaled_vertex = vertex * 3.0
            screen_pos = engine.project_3d_to_2d(
                scaled_vertex, model_matrix, view_matrix, projection_matrix
            )
            # Center on screen
            ##screen_pos.x += engine.width // 2
            ##screen_pos.y += engine.height // 2
            projected_vertices.append(screen_pos)
        
        # Draw edges with anti-aliased lines
        for i, edge in enumerate(icosa_edges):
            v1 = projected_vertices[edge[0]]
            v2 = projected_vertices[edge[1]]
            
            # Color based on edge index for visual appeal
            hue = (i * 137) % 360  # Golden ratio spacing
            r = int(127 + 127 * math.sin(math.radians(hue)))
            g = int(127 + 127 * math.sin(math.radians(hue + 120)))
            b = int(127 + 127 * math.sin(math.radians(hue + 240)))
            
            engine.draw_line_antialiased(
                v1.x, v1.y, v2.x, v2.y,
                '·', (r, g, b), (0, 0, 0)
            )
        
        # Draw vertices as small circles
        for i, vertex in enumerate(projected_vertices):
            # Different color for vertices
            hue = (i * 73) % 360  # Different spacing for variety
            r = int(200 + 55 * math.sin(math.radians(hue)))
            g = int(200 + 55 * math.sin(math.radians(hue + 120)))
            b = int(200 + 55 * math.sin(math.radians(hue + 240)))
            
            engine.draw_circle(
                int(vertex.x), int(vertex.y), 1,
                '•', (r, g, b), (0, 0, 0), filled=True
            )
        
        # Draw info text
        engine.draw_text(2, 2, "Icosahedron: 12 Vertices, 30 Edges", (255, 255, 255), (0, 0, 0))
        engine.draw_text(2, 4, "Use WASD to rotate manually", (200, 200, 200), (0, 0, 0))
        engine.draw_text(2, 5, "Press 'q' to return to menu", (200, 200, 200), (0, 0, 0))
        
        # Render frame
        engine.render()
        
        # Small delay
        time.sleep(0.02)


def run_sprites_transparency_demo(engine):
    """
    Demo 3: Sprites, Transparency & Blitting
    
    Demonstrates sprite drawing with transparency over a complex background.
    """
    last_time = time.time()
    sprite_x = engine.width // 2
    sprite_y = engine.height // 2
    sprite_dx = 1.5
    sprite_dy = 1.0
    
    print("Sprites, Transparency & Blitting Demo")
    print("Showing animated sprites with transparency over background")
    print("Press 'q' to return to menu")
    
    # Create background pattern (brick wall)
    brick_width = 8
    brick_height = 4
    brick_color = (120, 80, 60)
    mortar_color = (80, 50, 40)
    
    # Create a sprite (simple spaceship/character)
    sprite = Sprite(8, 8)
    
    # Define sprite pixels (None = transparent)
    sprite_pattern = [
        "  ████  ",
        " ██████ ",
        "████████",
        "█▒▒██▒▒█",
        "████████",
        " █ ██ █ ",
        "  █  █  ",
        "   ██   "
    ]
    
    sprite_colors = [
        (None, None, None, (100, 200, 255), (100, 200, 255), (100, 200, 255), None, None),
        (None, (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), None),
        ((100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255)),
        ((100, 200, 255), (255, 255, 100), (255, 255, 100), (100, 200, 255), (100, 200, 255), (255, 255, 100), (255, 255, 100), (100, 200, 255)),
        ((100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255), (100, 200, 255)),
        (None, (100, 200, 255), None, (100, 200, 255), (100, 200, 255), None, (100, 200, 255), None),
        (None, None, (100, 200, 255), None, None, (100, 200, 255), None, None),
        (None, None, None, (100, 200, 255), (100, 200, 255), None, None, None)
    ]
    
    # Populate sprite data
    for y in range(sprite.height):
        for x in range(sprite.width):
            char = sprite_pattern[y][x]
            fg_color = sprite_colors[y][x]
            if char != ' ' and fg_color is not None:  # Only set if not space and color exists
                sprite.set_pixel(x, y, char, fg_color, (0, 0, 0))
    
    while True:
        # Handle input
        key = engine.input_handler.get_key()
        if key == 'q':
            break
        
        # Calculate delta time
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        # Clear back buffer
        engine.clear()
        
        # Draw brick wall background
        for y in range(0, engine.height, brick_height):
            for x in range(0, engine.width, brick_width):
                # Alternate brick pattern
                offset = (y // brick_height) % 2 * (brick_width // 2)
                
                # Draw brick
                engine.draw_rectangle(
                    x + offset, y, 
                    brick_width - 1, brick_height - 1,
                    '▒', brick_color, mortar_color, filled=True
                )
                
                # Draw mortar lines
                if x + offset > 0:
                    for my in range(y, y + brick_height):
                        engine.set_pixel(x + offset, my, '│', mortar_color, mortar_color)
                if y > 0:
                    for mx in range(x + offset, x + offset + brick_width):
                        engine.set_pixel(mx, y, '─', mortar_color, mortar_color)
        
        # Update sprite position with bouncing
        sprite_x += sprite_dx
        sprite_y += sprite_dy
        
        # Bounce off walls
        if sprite_x <= 0 or sprite_x >= engine.width - sprite.width:
            sprite_dx = -sprite_dx
        if sprite_y <= 0 or sprite_y >= engine.height - sprite.height:
            sprite_dy = -sprite_dy
        
        # Draw multiple sprites with different colors
        for i in range(3):
            offset_x = int(math.sin(current_time * 2 + i * math.pi * 2 / 3) * 20)
            offset_y = int(math.cos(current_time * 2 + i * math.pi * 2 / 3) * 10)
            
            # Create color variant
            hue_shift = i * 40
            variant_sprite = Sprite(sprite.width, sprite.height)
            
            for y in range(sprite.height):
                for x in range(sprite.width):
                    original_cell = sprite.data[y][x]
                    if original_cell is not None and original_cell.fg is not None:  # Check for None
                        # Shift hue for variety
                        r, g, b = original_cell.fg
                        r = min(255, max(0, r + hue_shift))
                        g = min(255, max(0, g - hue_shift // 2))
                        b = min(255, max(0, b - hue_shift // 3))
                        variant_sprite.set_pixel(x, y, original_cell.char, (r, g, b), original_cell.bg)
            
            # Draw variant sprite
            engine.draw_sprite(
                int(sprite_x) + offset_x, 
                int(sprite_y) + offset_y, 
                variant_sprite
            )
        
        # Draw main sprite
        engine.draw_sprite(int(sprite_x), int(sprite_y), sprite)
        
        # Draw info text
        engine.draw_text(2, 2, "Sprite Transparency Demo", (255, 255, 255), (0, 0, 0))
        engine.draw_text(2, 4, "Background shows through transparent sprite pixels", (200, 200, 200), (0, 0, 0))
        engine.draw_text(2, 5, "Press 'q' to return to menu", (200, 200, 200), (0, 0, 0))
        
        # Render frame
        engine.render()
        
        # Small delay
        time.sleep(0.02)


def run_256_color_demo(engine):
    """
    Demo 4: 256-Color Palette & Text
    
    Demonstrates the 256-color palette with color cube and grayscale ramp.
    """
    print("256-Color Palette & Text Demo")
    print("Showing xterm 256-color palette with accurate RGB mapping")
    print("Press 'q' to return to menu")
    
    # Store original color mode
    original_color_mode = engine.use_true_color
    
    # Force 256-color mode for this demo
    engine.use_true_color = False
    
    try:
        while True:
            # Handle input
            key = engine.input_handler.get_key()
            if key == 'q':
                break
            
            # Clear back buffer
            engine.clear()
            
            # Draw title
            engine.draw_text(engine.width // 2 - 10, 1, "256-COLOR PALETTE", (255, 255, 255), (0, 0, 0))
            
            # Draw 6x6x6 color cube (16-231)
            cube_start_x = 2
            cube_start_y = 3
            cell_size = 2
            
            engine.draw_text(cube_start_x, cube_start_y - 1, "6x6x6 Color Cube (Colors 16-231)", (255, 255, 255), (0, 0, 0))
            
            for r in range(6):
                for g in range(6):
                    for b in range(6):
                        # Calculate RGB values
                        rgb_r = 0 if r == 0 else 40 + r * 40
                        rgb_g = 0 if g == 0 else 40 + g * 40
                        rgb_b = 0 if b == 0 else 40 + b * 40
                        
                        # Calculate position
                        x = cube_start_x + (r * 6 + g) * cell_size
                        y = cube_start_y + b
                        
                        # Draw color cell
                        engine.draw_rectangle(
                            x, y, cell_size, 1,
                            ' ', (rgb_r, rgb_g, rgb_b), (rgb_r, rgb_g, rgb_b), filled=True
                        )
            
            # Draw grayscale ramp (232-255)
            ramp_start_x = cube_start_x + 38 * cell_size
            ramp_start_y = cube_start_y
            
            engine.draw_text(ramp_start_x, ramp_start_y - 1, "Grayscale Ramp (232-255)", (255, 255, 255), (0, 0, 0))
            
            for i in range(24):
                gray_value = 8 + i * 10
                x = ramp_start_x + i * cell_size
                
                # Draw grayscale cell
                engine.draw_rectangle(
                    x, ramp_start_y, cell_size, 1,
                    ' ', (gray_value, gray_value, gray_value), (gray_value, gray_value, gray_value), filled=True
                )
                
                # Label every few cells
                if i % 4 == 0:
                    engine.draw_text(x, ramp_start_y + 2, str(232 + i), (255, 255, 255), (0, 0, 0))
            
            # Draw system colors (0-15)
            system_start_x = 2
            system_start_y = cube_start_y + 8
            
            engine.draw_text(system_start_x, system_start_y - 1, "System Colors (0-15)", (255, 255, 255), (0, 0, 0))
            
            # Standard system colors
            system_colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
                (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
                (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
            ]
            
            for i, color in enumerate(system_colors):
                x = system_start_x + i * 6
                
                # Draw color cell
                engine.draw_rectangle(
                    x, system_start_y, 4, 2,
                    ' ', color, color, filled=True
                )
                
                # Draw color index
                engine.draw_text(x + 1, system_start_y + 3, str(i), (255, 255, 255), (0, 0, 0))
            
            # Draw color mapping examples
            examples_start_y = system_start_y + 6
            
            engine.draw_text(system_start_x, examples_start_y, "Color Mapping Examples:", (255, 255, 255), (0, 0, 0))
            
            example_colors = [
                ("Red", (255, 0, 0)),
                ("Green", (0, 255, 0)),
                ("Blue", (0, 0, 255)),
                ("Cyan", (0, 255, 255)),
                ("Magenta", (255, 0, 255)),
                ("Yellow", (255, 255, 0)),
                ("Orange", (255, 165, 0)),
                ("Purple", (128, 0, 128)),
                ("Pink", (255, 192, 203)),
                ("Brown", (165, 42, 42))
            ]
            
            for i, (name, color) in enumerate(example_colors):
                x = system_start_x + (i % 5) * 20
                y = examples_start_y + 2 + (i // 5) * 3
                
                # Draw color sample
                engine.draw_rectangle(x, y, 8, 2, ' ', color, color, filled=True)
                
                # Draw color name and RGB values
                engine.draw_text(x, y - 1, name, (255, 255, 255), (0, 0, 0))
                engine.draw_text(x, y + 3, f"RGB{color}", (200, 200, 200), (0, 0, 0))
            
            # Draw info text
            engine.draw_text(2, engine.height - 3, "Color conversion uses intelligent RGB-to-xterm256 mapping", (200, 200, 200), (0, 0, 0))
            engine.draw_text(2, engine.height - 2, "Press 'q' to return to menu", (200, 200, 200), (0, 0, 0))
            
            # Render frame
            engine.render()
            
            # Small delay
            time.sleep(0.05)
    
    finally:
        # Restore original color mode
        engine.use_true_color = original_color_mode


def run_interactive_input_demo(engine):
    """
    Demo 5: Interactive Input Demo
    
    Demonstrates non-blocking input with an Etch-a-Sketch style drawing program.
    """
    cursor_x = engine.width // 2
    cursor_y = engine.height // 2
    drawing = False
    brush_char = '█'
    brush_color = (255, 255, 255)
    
    # Store the drawing canvas - we'll use the back buffer directly
    # but we need to preserve it between frames
    canvas = [[Cell() for _ in range(engine.width)] for _ in range(engine.height)]
    
    print("Interactive Input Demo (Etch-a-Sketch)")
    print("WASD: Move cursor | Space: Draw | C: Clear | Q: Return to menu")
    print("1-9: Change brush | R/G/B: Change color | E: Erase mode")
    
    while True:
        # Handle input
        key = engine.input_handler.get_key()
        if key is None:  # No key pressed
            pass
        elif key == 'q':
            break
        elif key == 'c':
            # Clear screen - reset canvas
            canvas = [[Cell() for _ in range(engine.width)] for _ in range(engine.height)]
        elif key == ' ':
            # Toggle drawing
            drawing = not drawing
        elif key == 'e':
            # Erase mode
            brush_char = ' '
            brush_color = (0, 0, 0)
        elif key in '123456789':
            # Change brush character
            brush_chars = ' ░▒▓█•·≡≣#'
            brush_char = brush_chars[int(key) - 1]
        elif key == 'r':
            brush_color = (255, 50, 50)  # Red
        elif key == 'g':
            brush_color = (50, 255, 50)  # Green
        elif key == 'b':
            brush_color = (50, 50, 255)  # Blue
        elif key == 'w' and cursor_y > 0:
            cursor_y -= 1
        elif key == 's' and cursor_y < engine.height - 1:
            cursor_y += 1
        elif key == 'a' and cursor_x > 0:
            cursor_x -= 1
        elif key == 'd' and cursor_x < engine.width - 1:
            cursor_x += 1

        elif key == 'u':
            brush_color = (255, 255, 255)  # white
        
        # Draw if in drawing mode
        if drawing:
            canvas[cursor_y][cursor_x] = Cell(brush_char, brush_color, (0, 0, 0))
        
        # Clear back buffer and draw the canvas
        engine.clear()
        
        # Copy canvas to back buffer
        for y in range(engine.height):
            for x in range(engine.width):
                engine.back_buffer[y][x] = canvas[y][x]
        
        # Draw cursor (invert colors for visibility)
        engine.set_pixel(cursor_x, cursor_y, 'X', (0, 0, 0), (255, 255, 255))
        
        # Draw UI
        engine.draw_text(2, 1, "ETCH-A-SKETCH", (255, 255, 0), (0, 0, 0))
        
        # Draw controls reference
        controls = [
            "WASD: Move",
            "Space: Draw",
            "C: Clear",
            "1-9: Brush",
            "R/G/B: Color",
            "E: Erase",
            "Q: Quit",
            "U: White color"
        ]
        
        for i, control in enumerate(controls):
            engine.draw_text(2, 3 + i, control, (200, 200, 200), (0, 0, 0))
        
        # Draw status
        status = f"Drawing: {'ON ' if drawing else 'OFF'} | Brush: '{brush_char}' | Color: RGB{brush_color}"
        engine.draw_text(2, engine.height - 2, status, (255, 255, 255), (0, 0, 0))
        
        # Render frame
        engine.render()
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.02)


def main():
    """Main function - runs the demo menu system."""
    # Create engine instance
    with TerminalGraphicsEngine(width=120, height=60, use_true_color=True) as engine:
        current_demo = None
        
        print("Terminal Graphics Engine Demo Suite")
        print("===================================")
        
        while True:
            # Clear screen and show menu
            engine.clear()
            
            # Draw menu title
            title = "TERMINAL GRAPHICS ENGINE DEMO SUITE"
            engine.draw_text(engine.width // 2 - len(title) // 2, 2, title, (255, 255, 100), (0, 0, 0))
            
            # Draw menu options
            menu_items = [
                "1. 2D Primitives & Anti-Aliasing",
                "2. 3D Math & Projection", 
                "3. Sprites & Transparency",
                "4. 256-Color Palette",
                "5. Interactive Input",
                "Q. Quit"
            ]
            
            menu_y = 6
            for item in menu_items:
                engine.draw_text(engine.width // 2 - 15, menu_y, item, (200, 200, 255), (0, 0, 0))
                menu_y += 2
            
            # Draw feature highlights
            features = [
                "• Flicker-free double buffering",
                "• 24-bit True Color & 256-color support", 
                "• Anti-aliased graphics",
                "• Cross-platform input handling",
                "• 2D/3D math library",
                "• Sprite blitting with transparency"
            ]
            
            features_y = menu_y + 2
            engine.draw_text(engine.width // 2 - 15, features_y - 1, "ENGINE FEATURES:", (255, 200, 100), (0, 0, 0))
            for feature in features:
                engine.draw_text(engine.width // 2 - 15, features_y, feature, (150, 200, 150), (0, 0, 0))
                features_y += 1
            
            # Draw footer
            footer = "Select a demo by pressing the corresponding number key"
            engine.draw_text(engine.width // 2 - len(footer) // 2, engine.height - 3, footer, (200, 200, 200), (0, 0, 0))
            
            # Render menu
            engine.render()
            
            # Handle menu input
            key = engine.input_handler.get_key()
            if key == 'q' or key == 'Q':
                break
            elif key == '1':
                current_demo = run_2d_primitives_demo
            elif key == '2':
                current_demo = run_3d_projection_demo
            elif key == '3':
                current_demo = run_sprites_transparency_demo
            elif key == '4':
                current_demo = run_256_color_demo
            elif key == '5':
                current_demo = run_interactive_input_demo
            
            # Run selected demo
            if current_demo:
                current_demo(engine)
                current_demo = None


if __name__ == "__main__":
    main()