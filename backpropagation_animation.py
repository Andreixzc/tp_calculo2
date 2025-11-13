from manim import *


class Backpropagation(Scene):
    """Backpropagation - rede horizontal com backprop indo da direita pra esquerda"""

    def construct(self):
        title = Text("Backpropagation", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Build horizontal network
        self.build_horizontal_network()
        self.wait(2)

        # Show backprop going backwards (right to left)
        self.show_backprop_backwards()
        self.wait(3)

    def build_horizontal_network(self):
        """Build simple horizontal network (left to right)"""

        # Neurons (horizontal layout)
        x = Circle(radius=0.3, color=BLUE, fill_opacity=0.3).shift(LEFT * 5)
        x_label = MathTex("x", font_size=28).move_to(x)

        h = Circle(radius=0.3, color=YELLOW, fill_opacity=0.2).shift(LEFT * 2)
        h_label = MathTex("h", font_size=28).move_to(h)

        y_hat = Circle(radius=0.3, color=GREEN, fill_opacity=0.3).shift(RIGHT * 2)
        y_label = MathTex(r"\hat{y}", font_size=28).move_to(y_hat)

        # Loss
        loss_box = Rectangle(height=0.6, width=1, color=RED, stroke_width=2).shift(
            RIGHT * 5
        )
        loss_label = MathTex("L", font_size=28, color=RED).move_to(loss_box)

        # Connections with weights
        w1_line = Line(x.get_right(), h.get_left(), color=GRAY, stroke_width=3)
        w1_label = MathTex("w_1", font_size=24, color=RED).next_to(
            w1_line, UP, buff=0.2
        )

        w2_line = Line(h.get_right(), y_hat.get_left(), color=GRAY, stroke_width=3)
        w2_label = MathTex("w_2", font_size=24, color=RED).next_to(
            w2_line, UP, buff=0.2
        )

        loss_line = Line(
            y_hat.get_right(), loss_box.get_left(), color=RED, stroke_width=2
        )

        # Animate forward pass
        self.play(Create(x), Write(x_label), run_time=1)
        self.wait(0.5)
        self.play(Create(w1_line), Write(w1_label), run_time=1.5)
        self.wait(0.5)
        self.play(Create(h), Write(h_label), run_time=1)
        self.wait(0.5)
        self.play(Create(w2_line), Write(w2_label), run_time=1.5)
        self.wait(0.5)
        self.play(Create(y_hat), Write(y_label), run_time=1)
        self.wait(0.5)
        self.play(Create(loss_line), run_time=1)
        self.wait(0.5)
        self.play(Create(loss_box), Write(loss_label), run_time=1.5)
        self.wait(1.5)

        self.neurons = VGroup(x, h, y_hat)
        self.labels = VGroup(x_label, h_label, y_label)
        self.lines = VGroup(w1_line, w2_line, loss_line)
        self.weight_labels = VGroup(w1_label, w2_label)
        self.loss_box = loss_box
        self.loss_label = loss_label

        self.network_group = VGroup(
            self.neurons,
            self.labels,
            self.lines,
            self.weight_labels,
            loss_box,
            loss_label,
        )

    def show_backprop_backwards(self):
        """Show backpropagation going from right to left"""

        # Move network up
        self.play(self.network_group.animate.shift(UP * 1.5), run_time=1.5)
        self.wait(1)

        # Title for backprop (correct arrow direction)
        backprop_title = Text(
            "Gradiente propagando para TRÁS ←",
            font_size=28,
            color=ORANGE,
            weight=BOLD,
        ).to_edge(DOWN, buff=2.8)

        self.play(Write(backprop_title), run_time=1.5)
        self.wait(1.5)

        # Fade out title before formulas
        self.play(FadeOut(backprop_title), run_time=0.8)
        self.wait(0.5)

        # Step 1: gradient at loss (start from right)
        step1_arrow = Arrow(
            self.loss_box.get_bottom() + DOWN * 0.2,
            self.loss_box.get_bottom() + DOWN * 1,
            color=ORANGE,
            stroke_width=3,
        )

        step1_formula = MathTex(
            r"\frac{\partial L}{\partial \hat{y}}", font_size=32, color=ORANGE
        ).next_to(step1_arrow, DOWN, buff=0.2)

        self.play(Create(step1_arrow), Write(step1_formula), run_time=2)
        self.wait(2)

        # Backward arrow from loss to y_hat
        back_arrow1 = Arrow(
            self.loss_box.get_left() + LEFT * 0.1,
            self.neurons[2].get_right() + RIGHT * 0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(Create(back_arrow1), run_time=1.5)
        self.wait(1)

        # Step 2: Update w2 (going backwards)
        step2_content = VGroup(
            MathTex(
                r"\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2}",
                font_size=26,
            ),
            MathTex(
                r"w_2 \leftarrow w_2 - \eta \cdot \frac{\partial L}{\partial w_2}",
                font_size=28,
                color=GREEN,
            ),
        ).arrange(DOWN, buff=0.4)
        step2_content.next_to(self.neurons[2], DOWN, buff=1.2)

        self.play(Write(step2_content[0]), run_time=2)
        self.wait(1.5)
        self.play(Write(step2_content[1]), run_time=2)
        self.wait(2)

        # Continue backwards: y_hat to h
        back_arrow2 = Arrow(
            self.neurons[2].get_left() + LEFT * 0.1,
            self.neurons[1].get_right() + RIGHT * 0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(FadeOut(step2_content), run_time=0.8)
        self.wait(0.3)
        self.play(Create(back_arrow2), run_time=1.5)
        self.wait(1)

        # Step 3: Update w1 (continue going backwards)
        step3_content = VGroup(
            MathTex(
                r"\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial w_1}",
                font_size=26,
            ),
            MathTex(
                r"w_1 \leftarrow w_1 - \eta \cdot \frac{\partial L}{\partial w_1}",
                font_size=28,
                color=GREEN,
            ),
        ).arrange(DOWN, buff=0.4)
        step3_content.next_to(self.neurons[1], DOWN, buff=1.2)

        self.play(Write(step3_content[0]), run_time=2)
        self.wait(1.5)
        self.play(Write(step3_content[1]), run_time=2)
        self.wait(2)

        # Final backwards arrow to input
        back_arrow3 = Arrow(
            self.neurons[1].get_left() + LEFT * 0.1,
            self.neurons[0].get_right() + RIGHT * 0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(Create(back_arrow3), run_time=1.5)
        self.wait(2.5)
