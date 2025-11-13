from manim import *
import numpy as np


class GradientDescent3D(Scene):
    """Gradient descent em 3D - superfície"""

    def construct(self):
        title = Text("Gradient Descent - 3D", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show 3D surface representation
        self.show_3d_concept()
        self.wait(3)

    def show_3d_concept(self):
        """Show 3D surface concept with 2D projection"""

        # Create axes for top-down view
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        )

        labels = VGroup(
            MathTex("w_1", font_size=32).next_to(
                axes.x_axis.get_end(), RIGHT, buff=0.2
            ),
            MathTex("w_2", font_size=32).next_to(
                axes.y_axis.get_end(), UP, buff=0.2
            ),
        )

        # Contour lines (representing height)
        contours = VGroup()
        for r in [0.3, 0.7, 1.1, 1.5, 1.9]:
            circle = Circle(
                radius=r * 1.5, color=BLUE, stroke_width=2, stroke_opacity=0.4
            )
            contours.add(circle)

        # Center (minimum)
        min_dot = Dot(ORIGIN, color=GREEN, radius=0.15)
        min_label = Text("Mínimo", font_size=28, color=GREEN).next_to(
            min_dot, DOWN, buff=0.4
        )

        # Label for contours
        contour_label = Text(
            "Curvas de nível\n(visão de cima)", font_size=24, slant=ITALIC
        ).to_corner(UL, buff=0.8)

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)
        self.play(Write(contour_label), run_time=1.5)
        self.wait(1)
        self.play(
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.3),
            run_time=2.5,
        )
        self.wait(1)
        self.play(Create(min_dot), Write(min_label), run_time=1.5)
        self.wait(1.5)

        # Starting point and descent path
        start_point = axes.c2p(1.5, 1.2)
        current_dot = Dot(start_point, color=RED, radius=0.12)

        self.play(Create(current_dot), run_time=1)
        self.wait(1)

        # Gradient descent path
        path_points = [
            (1.5, 1.2),
            (1.1, 0.85),
            (0.7, 0.5),
            (0.35, 0.2),
            (0.1, 0.05),
            (0, 0),
        ]

        gradient_text = (
            MathTex(
                r"\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{bmatrix}",
                font_size=32,
                color=YELLOW,
            )
            .to_edge(RIGHT, buff=0.8)
            .shift(UP * 1)
        )

        update_rule = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L",
            font_size=34,
            color=ORANGE,
        ).next_to(gradient_text, DOWN, buff=0.6)

        self.play(Write(gradient_text), run_time=2)
        self.wait(1.5)
        self.play(Write(update_rule), run_time=2)
        self.wait(1.5)

        # Animate descent
        for i in range(len(path_points) - 1):
            p1 = axes.c2p(*path_points[i])
            p2 = axes.c2p(*path_points[i + 1])

            arrow = Arrow(p1, p2, color=ORANGE, stroke_width=3, buff=0.1)
            new_dot = Dot(p2, color=RED, radius=0.12)

            self.play(Create(arrow), run_time=0.8)
            self.wait(0.3)
            self.play(
                Transform(current_dot, new_dot), FadeOut(arrow), run_time=1
            )
            self.wait(0.6)

        self.wait(2.5)
