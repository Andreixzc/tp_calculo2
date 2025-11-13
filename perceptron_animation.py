from manim import *
import numpy as np

# ==============================================================================
# 2 ANIMAÇÕES PRINCIPAIS:
# 1. PerceptronArchitecture - Estrutura + Regra de aprendizado
# 2. PerceptronLearningDemo - Processo de atualização dinâmica dos pesos
#
# Renderizar (teste rápido):
#   manim -pql perceptron_animation.py PerceptronArchitecture
#   manim -pql perceptron_animation.py PerceptronLearningDemo
#
# Renderizar (alta qualidade):
#   manim -pqh perceptron_animation.py PerceptronArchitecture
#   manim -pqh perceptron_animation.py PerceptronLearningDemo
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
                r"w_i \leftarrow w_i + \eta \cdot \text{erro} \cdot x_i",
                font_size=34,
                color=GREEN,
            ),
            MathTex(
                r"b \leftarrow b + \eta \cdot \text{erro}",
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


class PerceptronLearningDemo(Scene):
    """Demonstração do processo de atualização dinâmica dos pesos"""

    def construct(self):
        title = Text(
            "Perceptron: Atualização de Pesos",
            font_size=44,
            weight=BOLD,
            color=YELLOW,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(1)

        # Mock training examples
        self.training_examples = [
            ([0.2, 0.8], 1),
            ([0.9, 0.3], 1),
            ([0.1, 0.2], 0),
            ([0.7, 0.6], 1),
            ([0.3, 0.1], 0),
        ]

        # Initial weights and bias
        self.w1 = 0.5
        self.w2 = -0.3
        self.b = 0.2
        self.eta = 0.3  # learning rate (η)

        # Build visualization
        self.build_animated_perceptron()
        self.wait(1)

        # Show training process
        self.train_with_examples()

        self.wait(3)

    def build_animated_perceptron(self):
        """Build perceptron with animated components"""

        # Inputs
        self.x1_circle = Circle(radius=0.35, color=BLUE, fill_opacity=0.3).shift(
            LEFT * 4 + UP * 1
        )
        self.x2_circle = Circle(radius=0.35, color=BLUE, fill_opacity=0.3).shift(
            LEFT * 4 + DOWN * 1
        )

        self.x1_value = DecimalNumber(0, num_decimal_places=1, font_size=28).move_to(
            self.x1_circle
        )
        self.x2_value = DecimalNumber(0, num_decimal_places=1, font_size=28).move_to(
            self.x2_circle
        )

        x1_label = MathTex("x_1", font_size=28).next_to(
            self.x1_circle, LEFT, buff=0.2
        )
        x2_label = MathTex("x_2", font_size=28).next_to(
            self.x2_circle, LEFT, buff=0.2
        )

        # Neuron
        self.neuron = Circle(
            radius=0.6, color=YELLOW, fill_opacity=0.2, stroke_width=3
        )
        sigma = MathTex(r"\Sigma", font_size=48).move_to(self.neuron)

        # Weight lines
        self.w1_line = Line(
            self.x1_circle.get_right(),
            self.neuron.get_left() + UP * 0.25,
            color=GRAY,
            stroke_width=3,
        )
        self.w2_line = Line(
            self.x2_circle.get_right(),
            self.neuron.get_left() + DOWN * 0.25,
            color=GRAY,
            stroke_width=3,
        )

        # Weight labels
        self.w1_value = DecimalNumber(
            self.w1, num_decimal_places=2, font_size=26, color=RED
        )
        self.w1_value.add_updater(
            lambda m: m.next_to(self.w1_line, UP, buff=0.15)
        )

        self.w2_value = DecimalNumber(
            self.w2, num_decimal_places=2, font_size=26, color=RED
        )
        self.w2_value.add_updater(
            lambda m: m.next_to(self.w2_line, DOWN, buff=0.15)
        )

        # Bias
        bias_dot = Dot(color=PURPLE, radius=0.08).shift(DOWN * 2.5)
        self.bias_line = Line(
            bias_dot.get_top(), self.neuron.get_bottom(), color=PURPLE, stroke_width=3
        )

        self.b_value = DecimalNumber(
            self.b, num_decimal_places=2, font_size=26, color=PURPLE
        )
        self.b_value.next_to(self.bias_line, RIGHT, buff=0.2)

        # Output
        self.output_circle = Circle(
            radius=0.35, color=GREEN, fill_opacity=0.3
        ).shift(RIGHT * 3)
        self.y_value = DecimalNumber(0, num_decimal_places=0, font_size=28).move_to(
            self.output_circle
        )

        output_line = Line(
            self.neuron.get_right(),
            self.output_circle.get_left(),
            color=GREEN,
            stroke_width=3,
        )
        y_label = MathTex("y", font_size=28).next_to(
            self.output_circle, RIGHT, buff=0.2
        )

        self.perceptron_group = VGroup(
            self.x1_circle,
            self.x2_circle,
            self.x1_value,
            self.x2_value,
            x1_label,
            x2_label,
            self.w1_line,
            self.w2_line,
            self.w1_value,
            self.w2_value,
            self.neuron,
            sigma,
            bias_dot,
            self.bias_line,
            self.b_value,
            output_line,
            self.output_circle,
            self.y_value,
            y_label,
        )

        self.perceptron_group.scale(0.7).shift(UP * 0.5)

        self.play(Create(self.perceptron_group), run_time=2)

    def train_with_examples(self):
        """Show training process with mock examples"""

        # Example counter (no box)
        example_counter = Integer(0, font_size=28, color=ORANGE)
        counter_text = Text("Exemplo: ", font_size=26)
        counter_display = VGroup(counter_text, example_counter).arrange(RIGHT)
        counter_display.to_edge(RIGHT, buff=0.8).shift(UP * 2.5)
        self.play(Write(counter_display), run_time=0.8)

        for idx, (inputs, target) in enumerate(self.training_examples):
            x1, x2 = inputs

            # Update example counter
            self.play(example_counter.animate.set_value(idx + 1), run_time=0.3)

            # Show input values
            self.play(
                self.x1_value.animate.set_value(x1),
                self.x2_value.animate.set_value(x2),
                self.x1_circle.animate.set_fill(BLUE, opacity=0.2 + 0.5 * x1),
                self.x2_circle.animate.set_fill(BLUE, opacity=0.2 + 0.5 * x2),
                run_time=0.8,
            )

            # Highlight data flow
            self.play(
                self.w1_line.animate.set_color(YELLOW).set_stroke(width=5),
                self.w2_line.animate.set_color(YELLOW).set_stroke(width=5),
                run_time=0.4,
            )
            self.play(
                self.w1_line.animate.set_color(GRAY).set_stroke(width=3),
                self.w2_line.animate.set_color(GRAY).set_stroke(width=3),
                run_time=0.3,
            )

            # Calculate z
            z = self.w1 * x1 + self.w2 * x2 + self.b

            # Show calculation (positioned on the right side)
            calc_text = MathTex(
                f"z = {self.w1:.2f} \\cdot {x1:.1f} + {self.w2:.2f} \\cdot {x2:.1f} + {self.b:.2f}",
                font_size=22,
            )
            calc_text.to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

            z_result = MathTex(f"z = {z:.2f}", font_size=26, color=BLUE)
            z_result.next_to(calc_text, DOWN, buff=0.4)

            self.play(Write(calc_text), run_time=0.8)
            self.wait(0.3)
            self.play(Write(z_result), run_time=0.6)

            # Neuron activates
            self.play(
                self.neuron.animate.set_fill(YELLOW, opacity=min(abs(z) * 0.4, 0.5)),
                run_time=0.5,
            )

            # Calculate prediction
            y_pred = 1 if z >= 0 else 0

            # Show prediction
            pred_text = MathTex(
                f"y_{{pred}} = {y_pred}", font_size=26, color=GREEN
            )
            pred_text.next_to(z_result, DOWN, buff=0.4)
            self.play(Write(pred_text), run_time=0.6)

            # Output pulses
            self.play(
                self.output_circle.animate.scale(1.2).set_fill(
                    GREEN if y_pred == target else RED, opacity=0.6
                ),
                run_time=0.4,
            )
            self.play(
                self.output_circle.animate.scale(1 / 1.2).set_fill(GREEN, opacity=0.3),
                self.y_value.animate.set_value(y_pred),
                run_time=0.4,
            )

            # Calculate error
            error = target - y_pred
            error_color = GREEN if error == 0 else RED

            error_text = MathTex(
                f"\\text{{erro}} = {target} - {y_pred} = {error}",
                font_size=24,
                color=error_color,
            )
            error_text.next_to(pred_text, DOWN, buff=0.4)
            self.play(Write(error_text), run_time=0.8)
            self.wait(0.5)

            # Update weights if error
            if error != 0:
                # Calculate new weights
                new_w1 = self.w1 + self.eta * error * x1
                new_w2 = self.w2 + self.eta * error * x2
                new_b = self.b + self.eta * error

                # Show update animation
                self.play(
                    self.w1_value.animate.set_value(new_w1),
                    self.w2_value.animate.set_value(new_w2),
                    self.b_value.animate.set_value(new_b),
                    self.w1_line.animate.set_color(YELLOW),
                    self.w2_line.animate.set_color(YELLOW),
                    self.bias_line.animate.set_color(YELLOW),
                    run_time=1.2,
                )

                # Flash effect
                self.play(
                    self.w1_value.animate.scale(1.3),
                    self.w2_value.animate.scale(1.3),
                    self.b_value.animate.scale(1.3),
                    run_time=0.25,
                )
                self.play(
                    self.w1_value.animate.scale(1 / 1.3),
                    self.w2_value.animate.scale(1 / 1.3),
                    self.b_value.animate.scale(1 / 1.3),
                    run_time=0.25,
                )

                # Reset colors
                self.play(
                    self.w1_line.animate.set_color(GRAY),
                    self.w2_line.animate.set_color(GRAY),
                    self.bias_line.animate.set_color(PURPLE),
                    run_time=0.4,
                )

                self.w1 = new_w1
                self.w2 = new_w2
                self.b = new_b
            else:
                # Show checkmark
                check = Text("✓", font_size=40, color=GREEN, weight=BOLD)
                check.next_to(error_text, RIGHT, buff=0.3)
                self.play(Write(check), run_time=0.5)
                self.wait(0.5)
                self.play(FadeOut(check), run_time=0.3)

            # Clear info
            self.play(
                FadeOut(calc_text),
                FadeOut(z_result),
                FadeOut(pred_text),
                FadeOut(error_text),
                self.neuron.animate.set_fill(YELLOW, opacity=0.2),
                run_time=0.6,
            )
            self.wait(0.5)

        # Final message
        final_text = VGroup(
            Text("✓ Treinamento Completo!", font_size=32, color=GREEN, weight=BOLD),
            Text(
                f"Pesos finais: w₁={self.w1:.2f}, w₂={self.w2:.2f}, b={self.b:.2f}",
                font_size=24,
                color=YELLOW,
            ),
        ).arrange(DOWN, buff=0.4)
        final_text.to_edge(DOWN, buff=0.8)

        self.play(
            Write(final_text),
            self.neuron.animate.set_color(GREEN).set_stroke(width=4),
            run_time=1.5,
        )

        # Celebration pulse
        for _ in range(2):
            self.play(self.neuron.animate.scale(1.15), run_time=0.3)
            self.play(self.neuron.animate.scale(1 / 1.15), run_time=0.3)

        self.wait(1)


# ==============================================================================
# ==============================================================================
# DURAÇÃO APROXIMADA:
# - PerceptronArchitecture: ~45-50 segundos
# - PerceptronLearningDemo: ~50-60 segundos
# TOTAL: ~1:45 minutos
# ==============================================================================
