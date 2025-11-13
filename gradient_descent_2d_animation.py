from manim import *
import numpy as np


class GradientDescent2D(Scene):
    """Gradient descent em 2D - curva simples"""

    def construct(self):
        title = Text("Gradient Descent - 2D", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show curve
        self.show_curve()
        self.wait(2)

        # Show descent
        self.show_descent()
        self.wait(3)

    def show_curve(self):
        """Show loss curve"""

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 5, 1],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)

        labels = VGroup(
            MathTex("w", font_size=36).next_to(axes.x_axis.get_end(), RIGHT, buff=0.2),
            MathTex("L(w)", font_size=36).next_to(axes.y_axis.get_end(), UP, buff=0.2),
        )

        # Parabola
        curve = axes.plot(lambda x: 0.5 * x**2 + 0.5, color=BLUE, stroke_width=4)

        # Minimum point
        min_dot = Dot(axes.c2p(0, 0.5), color=GREEN, radius=0.1)
        min_label = Text("Mínimo", font_size=24, color=GREEN).next_to(
            min_dot, DOWN, buff=0.3
        )

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)
        self.play(Create(curve), run_time=2)
        self.wait(1)
        self.play(Create(min_dot), Write(min_label), run_time=1.5)
        self.wait(1.5)

        self.axes = axes
        self.curve = curve

    def show_descent(self):
        """Animate gradient descent steps"""

        # Starting point
        x_start = 2.5
        point = Dot(
            self.axes.c2p(x_start, 0.5 * x_start**2 + 0.5),
            color=RED,
            radius=0.12,
        )
        point_label = MathTex("w_0", font_size=28, color=RED).next_to(
            point, UP, buff=0.2
        )

        self.play(Create(point), Write(point_label), run_time=1.5)
        self.wait(1)

        # Show derivative - MOVED TO TOP to avoid overlap
        derivative_text = MathTex(
            r"\frac{dL}{dw} < 0 \Rightarrow \text{descer}",
            font_size=32,
            color=YELLOW,
        ).to_edge(UP, buff=3.5)

        lr_text = Text("Learning rate correto", font_size=26, color=GREEN).next_to(
            derivative_text, DOWN, buff=0.3
        )

        self.play(Write(derivative_text), run_time=1.5)
        self.wait(0.5)
        self.play(Write(lr_text), run_time=1)
        self.wait(1)

        # Gradient descent steps
        x_vals = [2.5, 1.8, 1.2, 0.7, 0.3, 0.1]

        for i, x_new in enumerate(x_vals[1:], 1):
            x_old = x_vals[i - 1]
            y_old = 0.5 * x_old**2 + 0.5
            y_new = 0.5 * x_new**2 + 0.5

            # Arrow showing step
            arrow = Arrow(
                self.axes.c2p(x_old, y_old),
                self.axes.c2p(x_new, y_new),
                color=ORANGE,
                stroke_width=3,
                buff=0.1,
            )

            new_point = Dot(self.axes.c2p(x_new, y_new), color=RED, radius=0.12)
            new_label = MathTex(f"w_{i}", font_size=28, color=RED).next_to(
                new_point, UP, buff=0.2
            )

            self.play(Create(arrow), run_time=0.8)
            self.wait(0.3)
            self.play(
                Transform(point, new_point),
                Transform(point_label, new_label),
                FadeOut(arrow),
                run_time=1,
            )
            self.wait(0.8)

        # Final message
        converged = Text("Convergiu!", font_size=32, color=GREEN, weight=BOLD).next_to(
            lr_text, DOWN, buff=0.3
        )

        self.play(Write(converged), run_time=1.5)
        self.wait(2)

        # Now show high learning rate problem
        self.play(
            FadeOut(point),
            FadeOut(point_label),
            FadeOut(converged),
            FadeOut(lr_text),
            FadeOut(derivative_text),
            run_time=1,
        )
        self.wait(0.5)

        self.show_overshooting()

    def show_overshooting(self):
        """Show what happens with too high learning rate"""

        # Warning text - MOVED TO TOP
        warning = Text(
            "Learning rate MUITO ALTO!", font_size=30, color=RED, weight=BOLD
        ).to_edge(UP, buff=3.5)
        self.play(Write(warning), run_time=1.5)
        self.wait(1)

        # Starting point
        x_start = 2
        point = Dot(
            self.axes.c2p(x_start, 0.5 * x_start**2 + 0.5),
            color=ORANGE,
            radius=0.12,
        )

        self.play(Create(point), run_time=1)
        self.wait(0.5)

        # Overshooting - bouncing back and forth WITH ROLLING ANIMATION
        x_vals = [2, -2.3, 2.5, -2.7, 2.9, -2.9]

        for i, x_new in enumerate(x_vals[1:], 1):
            x_old = x_vals[i - 1]
            y_old = 0.5 * x_old**2 + 0.5
            y_new = 0.5 * x_new**2 + 0.5

            arrow = Arrow(
                self.axes.c2p(x_old, y_old),
                self.axes.c2p(x_new, y_new),
                color=RED,
                stroke_width=3,
                buff=0.1,
            )

            # Create path for rolling animation
            num_steps = 15
            x_path = np.linspace(x_old, x_new, num_steps)

            self.play(Create(arrow), run_time=0.7)
            self.wait(0.2)

            # Animate point rolling along the curve
            for x_pos in x_path[1:]:
                y_pos = 0.5 * x_pos**2 + 0.5
                new_point = Dot(self.axes.c2p(x_pos, y_pos), color=ORANGE, radius=0.12)
                self.play(
                    Transform(point, new_point),
                    run_time=0.05,
                    rate_func=linear,
                )

            self.play(FadeOut(arrow), run_time=0.3)
            self.wait(0.3)

        diverged = Text(
            "Nunca converge!", font_size=32, color=RED, weight=BOLD
        ).next_to(warning, DOWN, buff=0.3)

        self.play(Write(diverged), run_time=1.5)
        self.wait(2.5)
