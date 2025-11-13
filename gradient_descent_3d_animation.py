from manim import *
import numpy as np


class GradientDescent3D(ThreeDScene):
    """Gradient descent em 3D - superfície tipo morro"""

    def construct(self):
        # Title
        title = Text("Gradient Descent - 3D", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=1.5)
        self.wait(1)

        # Show 3D surface representation
        self.show_3d_surface()
        self.wait(3)

    def show_3d_surface(self):
        """Show gradient descent on a 3D surface like a hill"""

        # Setup 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 5, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )

        # Axis labels
        x_label = MathTex("w_1", font_size=32).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("w_2", font_size=32).next_to(axes.y_axis.get_end(), UP)
        z_label = MathTex("L", font_size=32).next_to(axes.z_axis.get_end(), OUT)
        
        labels = VGroup(x_label, y_label, z_label)
        self.add_fixed_in_frame_mobjects(labels)

        # Loss function surface (irregular mountain-like terrain)
        def loss_function(u, v):
            # Complex surface with multiple peaks and valleys
            return (
                0.5 * (u**2 + v**2)  # Base paraboloid
                + 0.8 * np.sin(2 * u) * np.cos(2 * v)  # Waves creating irregularities
                + 0.3 * np.sin(3 * u + v)  # Additional complexity
                + 0.2 * np.cos(u - 2 * v)  # More irregular features
            )

        surface = Surface(
            lambda u, v: axes.c2p(u, v, loss_function(u, v)),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(40, 40),  # Higher resolution for irregular surface
            fill_opacity=0.75,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # Set camera
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.8)
        
        # Create axes and surface
        self.play(Create(axes), run_time=2)
        self.play(Write(labels), run_time=1)
        self.wait(1)
        
        self.play(Create(surface), run_time=3)
        self.wait(1)

        # Rotate camera to see the surface better
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        # Reset to good viewing angle
        self.move_camera(phi=70 * DEGREES, theta=-60 * DEGREES, run_time=2)
        self.wait(1)

        # Starting point (high on the irregular surface)
        start_x, start_y = 2.2, 1.8
        start_z = loss_function(start_x, start_y)
        
        # Gradient descent path points (navigating through irregular terrain)
        path_points = [
            (2.2, 1.8),   # Start high
            (1.9, 1.5),   # Descending
            (1.5, 1.2),   # Through first valley
            (1.2, 0.8),   # Around a peak
            (0.9, 0.5),   # Navigating irregularities
            (0.6, 0.3),   # Getting closer
            (0.35, 0.15), # Almost there
            (0.15, 0.05), # Final approach
            (0.0, 0.0),   # Global minimum
        ]

        # Create path on surface
        path_dots = []
        path_lines = []
        
        for i, (x, y) in enumerate(path_points):
            z = loss_function(x, y)
            dot = Sphere(radius=0.08, color=RED if i > 0 else YELLOW)
            dot.move_to(axes.c2p(x, y, z))
            path_dots.append(dot)

        # Show formulas (fixed to camera)
        gradient_text = MathTex(
            r"\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{bmatrix}",
            font_size=28,
            color=YELLOW,
        ).to_corner(UR, buff=0.5).shift(DOWN * 0.5)

        update_rule = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L",
            font_size=30,
            color=ORANGE,
        ).next_to(gradient_text, DOWN, buff=0.4)

        self.add_fixed_in_frame_mobjects(gradient_text, update_rule)
        self.play(Write(gradient_text), run_time=1.5)
        self.wait(1)
        self.play(Write(update_rule), run_time=1.5)
        self.wait(1)

        # Show starting point
        self.play(Create(path_dots[0]), run_time=1)
        self.wait(1)

        # Animate gradient descent
        for i in range(len(path_dots) - 1):
            # Draw line connecting points
            line = Line3D(
                start=path_dots[i].get_center(),
                end=path_dots[i + 1].get_center(),
                color=ORANGE,
                thickness=0.02,
            )
            
            self.play(Create(line), run_time=0.5)
            self.play(Create(path_dots[i + 1]), run_time=0.6)
            self.wait(0.4)

        # Highlight minimum
        min_dot = path_dots[-1]
        self.play(
            min_dot.animate.set_color(GREEN).scale(1.5),
            run_time=1
        )
        
        # Pulse effect
        for _ in range(2):
            self.play(min_dot.animate.scale(1.2), run_time=0.3)
            self.play(min_dot.animate.scale(1/1.2), run_time=0.3)
        
        self.wait(1)

        # Final rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
