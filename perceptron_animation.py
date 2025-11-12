from manim import *
import numpy as np

# ==============================================================================
# 3 ANIMAÇÕES PRINCIPAIS:
# 1. PerceptronArchitecture - Estrutura + Regra de aprendizado
# 2. ANDSuccess - Perceptron resolve AND (linearmente separável)
# 3. XORFailure - Perceptron falha com XOR (não-linearmente separável)
#
# Renderizar (teste rápido):
#   manim -pql perceptron_animation.py PerceptronArchitecture
#   manim -pql perceptron_animation.py ANDSuccess
#   manim -pql perceptron_animation.py XORFailure
#
# Renderizar (alta qualidade, 60fps):
#   manim -pqh perceptron_animation.py PerceptronArchitecture
#   manim -pqh perceptron_animation.py ANDSuccess
#   manim -pqh perceptron_animation.py XORFailure
# ==============================================================================


class PerceptronArchitecture(Scene):
    """Arquitetura do perceptron original + regra de aprendizado"""

    def construct(self):
        title = Text("O Perceptron (1958)", font_size=48, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show structure
        self.build_perceptron()
        self.wait(2)

        # Show learning rule
        self.show_learning_rule()
        self.wait(3)

    def build_perceptron(self):
        """Build perceptron diagram"""

        # Inputs
        x1 = Circle(radius=0.3, color=BLUE, fill_opacity=0.3).shift(
            LEFT * 5 + UP * 1.2
        )
        x2 = Circle(radius=0.3, color=BLUE, fill_opacity=0.3).shift(
            LEFT * 5 + DOWN * 1.2
        )

        x1_label = MathTex("x_1", font_size=36).next_to(x1, LEFT, buff=0.3)
        x2_label = MathTex("x_2", font_size=36).next_to(x2, LEFT, buff=0.3)

        # Neuron
        neuron = Circle(
            radius=0.8, color=YELLOW, fill_opacity=0.2, stroke_width=3
        )
        sigma = MathTex(r"\Sigma", font_size=60).move_to(neuron)

        # Weights
        w1_line = Line(
            x1.get_right(),
            neuron.get_left() + UP * 0.35,
            color=GRAY,
            stroke_width=2,
        )
        w2_line = Line(
            x2.get_right(),
            neuron.get_left() + DOWN * 0.35,
            color=GRAY,
            stroke_width=2,
        )

        w1_label = MathTex("w_1", font_size=32, color=RED).next_to(
            w1_line, UP, buff=0.2
        )
        w2_label = MathTex("w_2", font_size=32, color=RED).next_to(
            w2_line, DOWN, buff=0.2
        )

        # Bias
        bias = Dot(color=PURPLE, radius=0.08).shift(DOWN * 2.8)
        bias_label = MathTex("b", font_size=36, color=PURPLE).next_to(
            bias, DOWN, buff=0.2
        )
        bias_line = Line(
            bias.get_top(), neuron.get_bottom(), color=PURPLE, stroke_width=2
        )

        # Step function
        step_box = Rectangle(
            height=1.3, width=1.8, color=ORANGE, stroke_width=3
        ).shift(RIGHT * 3)
        step_text = Text("step", font_size=32, color=ORANGE).move_to(step_box)
        step_line = Line(
            neuron.get_right(),
            step_box.get_left(),
            color=ORANGE,
            stroke_width=2,
        )

        # Output
        output = Circle(radius=0.3, color=GREEN, fill_opacity=0.3).shift(
            RIGHT * 5.5
        )
        y_label = MathTex("y", font_size=36).next_to(output, RIGHT, buff=0.3)
        output_line = Line(
            step_box.get_right(),
            output.get_left(),
            color=GREEN,
            stroke_width=2,
        )

        # Animate construction
        self.play(
            LaggedStart(
                Create(x1),
                Create(x2),
                Write(x1_label),
                Write(x2_label),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait(1)

        self.play(
            Create(w1_line),
            Create(w2_line),
            Write(w1_label),
            Write(w2_label),
            run_time=2,
        )
        self.wait(1)

        self.play(Create(neuron), Write(sigma), run_time=1.5)
        self.wait(0.5)

        self.play(
            Create(bias), Write(bias_label), Create(bias_line), run_time=1.5
        )
        self.wait(1)

        self.play(
            Create(step_line), Create(step_box), Write(step_text), run_time=1.5
        )
        self.wait(0.5)

        self.play(
            Create(output_line), Create(output), Write(y_label), run_time=1.5
        )
        self.wait(1)

        self.perceptron = VGroup(
            x1,
            x2,
            x1_label,
            x2_label,
            w1_line,
            w2_line,
            w1_label,
            w2_label,
            neuron,
            sigma,
            bias,
            bias_label,
            bias_line,
            step_line,
            step_box,
            step_text,
            output_line,
            output,
            y_label,
        )

    def show_learning_rule(self):
        """Show mathematical formulas and learning rule"""

        # Scale and move perceptron
        self.play(
            self.perceptron.animate.scale(0.5).to_edge(LEFT, buff=0.8),
            run_time=2,
        )
        self.wait(1)

        # Formulas
        formulas = VGroup(
            Text("Matemática:", font_size=32, weight=BOLD, color=YELLOW),
            MathTex(r"z = w_1 x_1 + w_2 x_2 + b", font_size=38),
            MathTex(
                r"y = \begin{cases} 1 & \text{se } z \geq 0 \\ 0 & \text{se } z < 0 \end{cases}",
                font_size=36,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        formulas.to_edge(RIGHT, buff=1.5).shift(UP * 1.5)

        self.play(Write(formulas[0]), run_time=1)
        self.wait(1)
        self.play(Write(formulas[1]), run_time=1.5)
        self.wait(2)
        self.play(Write(formulas[2]), run_time=2)
        self.wait(2)

        # Learning rule
        learning = VGroup(
            Text("Aprendizado:", font_size=32, weight=BOLD, color=GREEN),
            MathTex(
                r"\text{erro} = y_{\text{real}} - y_{\text{predito}}",
                font_size=32,
                color=RED,
            ),
            MathTex(
                r"w_i \leftarrow w_i + \alpha \cdot \text{erro} \cdot x_i",
                font_size=34,
                color=GREEN,
            ),
            MathTex(
                r"b \leftarrow b + \alpha \cdot \text{erro}",
                font_size=34,
                color=GREEN,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        learning.next_to(formulas, DOWN, buff=1)

        self.play(Write(learning[0]), run_time=1)
        self.wait(1)
        self.play(Write(learning[1]), run_time=1.5)
        self.wait(2)
        self.play(Write(learning[2]), run_time=1.5)
        self.wait(2)
        self.play(Write(learning[3]), run_time=1.5)
        self.wait(2.5)


class ANDSuccess(Scene):
    """Perceptron resolve AND - linearmente separável"""

    def construct(self):
        title = Text("AND: Sucesso ✓", font_size=48, weight=BOLD, color=GREEN)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Truth table
        self.show_truth_table()
        self.wait(2)

        # 2D visualization
        self.show_2d_plot()
        self.wait(3)

    def show_truth_table(self):
        """Show AND truth table"""

        table_title = Text("Tabela verdade:", font_size=30, weight=BOLD)
        table_title.shift(UP * 2.3 + LEFT * 4.8)

        # Create table manually for better control
        table_data = VGroup(
            VGroup(
                MathTex("x_1", font_size=32),
                MathTex("x_2", font_size=32),
                MathTex("y", font_size=32),
            ).arrange(RIGHT, buff=0.8),
            VGroup(
                MathTex("0", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("0", font_size=30, color=RED),
                MathTex("1", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("1", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("1", font_size=30, color=GREEN),
                MathTex("1", font_size=30, color=GREEN),
                MathTex("1", font_size=30, color=GREEN),
            ).arrange(RIGHT, buff=1.0),
        ).arrange(DOWN, buff=0.4)

        table_data.next_to(table_title, DOWN, buff=0.4)

        # Box around table
        table_box = SurroundingRectangle(table_data, color=WHITE, buff=0.3)

        self.play(Write(table_title), run_time=1)
        self.wait(0.5)
        self.play(Create(table_box), run_time=1)
        self.wait(0.5)

        for row in table_data:
            self.play(Write(row), run_time=1)
            self.wait(0.8)

        self.wait(1)
        self.table_group = VGroup(table_title, table_box, table_data)

    def show_2d_plot(self):
        """Show 2D plot with decision boundary"""

        # Move table
        self.play(
            self.table_group.animate.scale(0.8).shift(LEFT * 0.3), run_time=1.5
        )
        self.wait(0.5)

        # Create axes
        axes = Axes(
            x_range=[-0.3, 1.3, 0.5],
            y_range=[-0.3, 1.3, 0.5],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True},
        ).shift(RIGHT * 2.8 + DOWN * 0.5)

        labels = VGroup(
            MathTex("x_1", font_size=36).next_to(
                axes.x_axis.get_end(), RIGHT, buff=0.2
            ),
            MathTex("x_2", font_size=36).next_to(
                axes.y_axis.get_end(), UP, buff=0.2
            ),
        )

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)

        # Plot points
        points = [
            (0, 0, RED, "(0,0) \\to 0"),
            (0, 1, RED, "(0,1) \\to 0"),
            (1, 0, RED, "(1,0) \\to 0"),
            (1, 1, GREEN, "(1,1) \\to 1"),
        ]

        dots = VGroup()
        point_labels = VGroup()

        for x, y, color, label_text in points:
            dot = Dot(axes.c2p(x, y), color=color, radius=0.15)
            label = MathTex(label_text, font_size=24, color=color)

            # Position labels to avoid overlap
            if x == 0 and y == 0:
                label.next_to(dot, DOWN + LEFT, buff=0.15)
            elif x == 0 and y == 1:
                label.next_to(dot, UP + LEFT, buff=0.15)
            elif x == 1 and y == 0:
                label.next_to(dot, DOWN + RIGHT, buff=0.15)
            else:
                label.next_to(dot, UP + RIGHT, buff=0.15)

            dots.add(dot)
            point_labels.add(label)

        self.play(
            LaggedStart(*[Create(d) for d in dots], lag_ratio=0.3), run_time=2
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*[Write(l) for l in point_labels], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(2)

        # Decision boundary
        separation_text = Text(
            "Linearmente separável!", font_size=34, weight=BOLD, color=YELLOW
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(separation_text), run_time=1.5)
        self.wait(1)

        # Draw line that correctly separates AND points
        # Line: x1 + x2 = 1.5 (separates (1,1) from the other three points)
        # Extended to be clearly visible
        line = Line(
            axes.c2p(0.3, 1.2), axes.c2p(1.2, 0.3), color=BLUE, stroke_width=5
        )

        self.play(Create(line), run_time=2)
        self.wait(2.5)


class XORFailure(Scene):
    """Perceptron falha com XOR - não-linearmente separável"""

    def construct(self):
        title = Text("XOR: Impossível ✗", font_size=48, weight=BOLD, color=RED)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Truth table
        self.show_truth_table()
        self.wait(2)

        # 2D visualization
        self.show_2d_plot()
        self.wait(2)

        # Try to separate (and fail)
        self.show_impossibility()
        self.wait(3)

    def show_truth_table(self):
        """Show XOR truth table"""

        table_title = Text("Tabela verdade:", font_size=30, weight=BOLD)
        table_title.shift(UP * 2.3 + LEFT * 4.8)

        table_data = VGroup(
            VGroup(
                MathTex("x_1", font_size=32),
                MathTex("x_2", font_size=32),
                MathTex("y", font_size=32),
            ).arrange(RIGHT, buff=0.8),
            VGroup(
                MathTex("0", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("0", font_size=30, color=GREEN),
                MathTex("1", font_size=30, color=GREEN),
                MathTex("1", font_size=30, color=GREEN),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("1", font_size=30, color=GREEN),
                MathTex("0", font_size=30, color=GREEN),
                MathTex("1", font_size=30, color=GREEN),
            ).arrange(RIGHT, buff=1.0),
            VGroup(
                MathTex("1", font_size=30, color=RED),
                MathTex("1", font_size=30, color=RED),
                MathTex("0", font_size=30, color=RED),
            ).arrange(RIGHT, buff=1.0),
        ).arrange(DOWN, buff=0.4)

        table_data.next_to(table_title, DOWN, buff=0.4)

        table_box = SurroundingRectangle(table_data, color=WHITE, buff=0.3)

        self.play(Write(table_title), run_time=1)
        self.wait(0.5)
        self.play(Create(table_box), run_time=1)
        self.wait(0.5)

        for row in table_data:
            self.play(Write(row), run_time=1)
            self.wait(0.8)

        self.wait(1)
        self.table_group = VGroup(table_title, table_box, table_data)

    def show_2d_plot(self):
        """Show 2D plot with XOR points"""

        # Move table
        self.play(
            self.table_group.animate.scale(0.8).shift(LEFT * 0.3), run_time=1.5
        )
        self.wait(0.5)

        # Create axes
        axes = Axes(
            x_range=[-0.3, 1.3, 0.5],
            y_range=[-0.3, 1.3, 0.5],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True},
        ).shift(RIGHT * 2.8 + DOWN * 0.5)

        labels = VGroup(
            MathTex("x_1", font_size=36).next_to(
                axes.x_axis.get_end(), RIGHT, buff=0.2
            ),
            MathTex("x_2", font_size=36).next_to(
                axes.y_axis.get_end(), UP, buff=0.2
            ),
        )

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)

        # Plot XOR points
        points = [
            (0, 0, RED, "(0,0) \\to 0"),
            (0, 1, GREEN, "(0,1) \\to 1"),
            (1, 0, GREEN, "(1,0) \\to 1"),
            (1, 1, RED, "(1,1) \\to 0"),
        ]

        dots = VGroup()
        point_labels = VGroup()

        for x, y, color, label_text in points:
            dot = Dot(axes.c2p(x, y), color=color, radius=0.15)
            label = MathTex(label_text, font_size=24, color=color)

            if x == 0 and y == 0:
                label.next_to(dot, DOWN + LEFT, buff=0.15)
            elif x == 0 and y == 1:
                label.next_to(dot, UP + LEFT, buff=0.15)
            elif x == 1 and y == 0:
                label.next_to(dot, DOWN + RIGHT, buff=0.15)
            else:
                label.next_to(dot, UP + RIGHT, buff=0.15)

            dots.add(dot)
            point_labels.add(label)

        self.play(
            LaggedStart(*[Create(d) for d in dots], lag_ratio=0.3), run_time=2
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*[Write(l) for l in point_labels], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(2)

        self.axes = axes
        self.dots = dots

    def show_impossibility(self):
        """Try different lines and show it's impossible"""

        impossibility_text = Text(
            "NÃO linearmente separável!", font_size=34, weight=BOLD, color=RED
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(impossibility_text), run_time=1.5)
        self.wait(1.5)

        # Try multiple lines and show they all fail
        line1 = Line(
            self.axes.c2p(-0.2, 0.5),
            self.axes.c2p(1.2, 0.5),
            color=ORANGE,
            stroke_width=4,
        )

        line2 = Line(
            self.axes.c2p(0.5, -0.2),
            self.axes.c2p(0.5, 1.2),
            color=ORANGE,
            stroke_width=4,
        )

        line3 = Line(
            self.axes.c2p(-0.2, 1.2),
            self.axes.c2p(1.2, -0.2),
            color=ORANGE,
            stroke_width=4,
        )

        cross = VGroup(
            Line(
                ORIGIN + UP * 0.15 + LEFT * 0.15,
                ORIGIN + DOWN * 0.15 + RIGHT * 0.15,
                color=RED,
                stroke_width=5,
            ),
            Line(
                ORIGIN + UP * 0.15 + RIGHT * 0.15,
                ORIGIN + DOWN * 0.15 + LEFT * 0.15,
                color=RED,
                stroke_width=5,
            ),
        )

        # Try line 1
        self.play(Create(line1), run_time=1.5)
        self.wait(1)
        cross1 = cross.copy().scale(0.8).next_to(line1, RIGHT, buff=0.3)
        self.play(Create(cross1), run_time=0.8)
        self.wait(1)
        self.play(FadeOut(line1), FadeOut(cross1), run_time=0.8)

        # Try line 2
        self.play(Create(line2), run_time=1.5)
        self.wait(1)
        cross2 = cross.copy().scale(0.8).next_to(line2, UP, buff=0.3)
        self.play(Create(cross2), run_time=0.8)
        self.wait(1)
        self.play(FadeOut(line2), FadeOut(cross2), run_time=0.8)

        # Try line 3
        self.play(Create(line3), run_time=1.5)
        self.wait(1)
        cross3 = cross.copy().scale(0.8).move_to(self.axes.c2p(0.8, 0.8))
        self.play(Create(cross3), run_time=0.8)
        self.wait(1.5)

        # Final message
        solution_text = Text(
            "Solução: Redes Multicamadas!",
            font_size=30,
            weight=BOLD,
            color=YELLOW,
        ).next_to(impossibility_text, UP, buff=0.5)

        self.play(Write(solution_text), run_time=1.5)
        self.wait(3)


# ==============================================================================
# DURAÇÃO APROXIMADA:
# - PerceptronArchitecture: ~45-50 segundos
# - ANDSuccess: ~35-40 segundos
# - XORFailure: ~45-50 segundos
# TOTAL: ~2 minutos
# ==============================================================================
