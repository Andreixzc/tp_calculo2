from manim import *
import numpy as np


class Backpropagation(Scene):
    """Backpropagation - rede horizontal com backprop indo da direita pra esquerda"""

    def construct(self):
        title = Text("Backpropagation", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Build horizontal network with bias
        self.build_horizontal_network()
        self.wait(2)

        # Show backprop going backwards (right to left)
        self.show_backprop_backwards()
        self.wait(2)
        
        # Clear and show complex network with gradient vectors
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.wait(0.5)
        
        # Show complex architecture with gradient vectors
        self.show_complex_gradient_architecture()
        self.wait(3)

    def build_horizontal_network(self):
        """Build simple horizontal network (left to right) with bias"""

        # Neurons (horizontal layout)
        x = Circle(radius=0.3, color=BLUE, fill_opacity=0.3).shift(LEFT * 5)
        x_label = MathTex("x", font_size=28).move_to(x)

        h = Circle(radius=0.3, color=YELLOW, fill_opacity=0.2).shift(LEFT * 2)
        h_label = MathTex("h", font_size=28).move_to(h)

        y_hat = Circle(radius=0.3, color=GREEN, fill_opacity=0.3).shift(
            RIGHT * 2
        )
        y_label = MathTex(r"\hat{y}", font_size=28).move_to(y_hat)

        # Loss
        loss_box = Rectangle(
            height=0.6, width=1, color=RED, stroke_width=2
        ).shift(RIGHT * 5)
        loss_label = MathTex("L", font_size=28, color=RED).move_to(loss_box)

        # Connections with weights
        w1_line = Line(x.get_right(), h.get_left(), color=GRAY, stroke_width=3)
        w1_label = MathTex("w_1", font_size=24, color=RED).next_to(
            w1_line, UP, buff=0.2
        )

        w2_line = Line(
            h.get_right(), y_hat.get_left(), color=GRAY, stroke_width=3
        )
        w2_label = MathTex("w_2", font_size=24, color=RED).next_to(
            w2_line, UP, buff=0.2
        )
        
        # Bias nodes (small squares)
        b1_square = Square(side_length=0.25, color=PURPLE, fill_opacity=0.3).shift(LEFT * 2 + UP * 1.2)
        b1_label = MathTex("b_1", font_size=20, color=PURPLE).move_to(b1_square)
        b1_line = Line(b1_square.get_bottom(), h.get_top(), color=PURPLE, stroke_width=2)
        
        b2_square = Square(side_length=0.25, color=PURPLE, fill_opacity=0.3).shift(RIGHT * 2 + UP * 1.2)
        b2_label = MathTex("b_2", font_size=20, color=PURPLE).move_to(b2_square)
        b2_line = Line(b2_square.get_bottom(), y_hat.get_top(), color=PURPLE, stroke_width=2)

        loss_line = Line(
            y_hat.get_right(), loss_box.get_left(), color=RED, stroke_width=2
        )

        # Animate forward pass
        self.play(Create(x), Write(x_label), run_time=0.8)
        self.wait(0.3)
        self.play(Create(w1_line), Write(w1_label), run_time=1)
        self.wait(0.3)
        self.play(Create(h), Write(h_label), run_time=0.8)
        self.play(Create(b1_line), Create(b1_square), Write(b1_label), run_time=0.8)
        self.wait(0.3)
        self.play(Create(w2_line), Write(w2_label), run_time=1)
        self.wait(0.3)
        self.play(Create(y_hat), Write(y_label), run_time=0.8)
        self.play(Create(b2_line), Create(b2_square), Write(b2_label), run_time=0.8)
        self.wait(0.3)
        self.play(Create(loss_line), run_time=0.8)
        self.wait(0.3)
        self.play(Create(loss_box), Write(loss_label), run_time=1)
        self.wait(1)

        self.neurons = VGroup(x, h, y_hat)
        self.labels = VGroup(x_label, h_label, y_label)
        self.lines = VGroup(w1_line, w2_line, loss_line)
        self.weight_labels = VGroup(w1_label, w2_label)
        self.bias_nodes = VGroup(b1_square, b2_square)
        self.bias_labels = VGroup(b1_label, b2_label)
        self.bias_lines = VGroup(b1_line, b2_line)
        self.loss_box = loss_box
        self.loss_label = loss_label
        
        # Store specific elements for alignment
        self.w1_line = w1_line
        self.w2_line = w2_line
        self.b1_line = b1_line
        self.b2_line = b2_line
        self.h_neuron = h
        self.y_neuron = y_hat

        self.network_group = VGroup(
            self.neurons,
            self.labels,
            self.lines,
            self.weight_labels,
            self.bias_nodes,
            self.bias_labels,
            self.bias_lines,
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

        # Step 2: Update w2 (aligned with w2 edge)
        step2_gradient = MathTex(
            r"\frac{\partial L}{\partial w_2} = \left( \frac{\partial L}{\partial \hat{y}} \right) \cdot \left( \frac{\partial \hat{y}}{\partial w_2} \right)",
            font_size=22,
        )
        # Position near w2 line
        step2_gradient.next_to(self.w2_line, DOWN, buff=0.3)

        step2_update = MathTex(
            r"w_2 \leftarrow w_2 - \left( \eta \cdot \frac{\partial L}{\partial w_2} \right)",
            font_size=24,
            color=GREEN,
        )
        step2_update.next_to(step2_gradient, DOWN, buff=0.3)

        self.play(Write(step2_gradient), run_time=1.5)
        self.wait(1)
        self.play(Write(step2_update), run_time=1.5)
        
        # Blink effect on w2 label
        self.play(
            self.weight_labels[1].animate.set_color(YELLOW).scale(1.3),
            run_time=0.3
        )
        self.play(
            self.weight_labels[1].animate.set_color(RED).scale(1/1.3),
            run_time=0.3
        )
        self.wait(1)
        
        # Step 2b: Update b2 (positioned next to weight formulas)
        step2b_gradient = MathTex(
            r"\frac{\partial L}{\partial b_2}",
            font_size=24,
            color=PURPLE,
        )
        # Position below the w2 formulas
        step2b_gradient.next_to(step2_update, DOWN, buff=0.5)

        step2b_update = MathTex(
            r"b_2 \leftarrow b_2 - \left( \eta \cdot \frac{\partial L}{\partial b_2} \right)",
            font_size=22,
            color=PURPLE,
        )
        step2b_update.next_to(step2b_gradient, DOWN, buff=0.2)

        self.play(Write(step2b_gradient), run_time=1)
        self.wait(0.5)
        self.play(Write(step2b_update), run_time=1.5)
        
        # Blink effect on b2 label
        self.play(
            self.bias_labels[1].animate.set_color(YELLOW).scale(1.3),
            run_time=0.3
        )
        self.play(
            self.bias_labels[1].animate.set_color(PURPLE).scale(1/1.3),
            run_time=0.3
        )
        self.wait(1)

        # Continue backwards: y_hat to h
        back_arrow2 = Arrow(
            self.neurons[2].get_left() + LEFT * 0.1,
            self.neurons[1].get_right() + RIGHT * 0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(
            FadeOut(step2_gradient), 
            FadeOut(step2_update),
            FadeOut(step2b_gradient),
            FadeOut(step2b_update),
            run_time=0.8
        )
        self.wait(0.3)
        self.play(Create(back_arrow2), run_time=1.5)
        self.wait(1)

        # Step 3: Update w1 (aligned with w1 edge)
        step3_gradient = MathTex(
            r"\frac{\partial L}{\partial w_1} = \left( \frac{\partial L}{\partial h} \right) \cdot \left( \frac{\partial h}{\partial w_1} \right)",
            font_size=22,
        )
        step3_gradient.next_to(self.w1_line, DOWN, buff=0.3)

        step3_update = MathTex(
            r"w_1 \leftarrow w_1 - \left( \eta \cdot \frac{\partial L}{\partial w_1} \right)",
            font_size=24,
            color=GREEN,
        )
        step3_update.next_to(step3_gradient, DOWN, buff=0.3)

        self.play(Write(step3_gradient), run_time=1.5)
        self.wait(1)
        self.play(Write(step3_update), run_time=1.5)
        
        # Blink effect on w1 label
        self.play(
            self.weight_labels[0].animate.set_color(YELLOW).scale(1.3),
            run_time=0.3
        )
        self.play(
            self.weight_labels[0].animate.set_color(RED).scale(1/1.3),
            run_time=0.3
        )
        self.wait(1)
        
        # Step 3b: Update b1 (positioned next to weight formulas)
        step3b_gradient = MathTex(
            r"\frac{\partial L}{\partial b_1}",
            font_size=24,
            color=PURPLE,
        )
        # Position below the w1 formulas
        step3b_gradient.next_to(step3_update, DOWN, buff=0.5)

        step3b_update = MathTex(
            r"b_1 \leftarrow b_1 - \left( \eta \cdot \frac{\partial L}{\partial b_1} \right)",
            font_size=22,
            color=PURPLE,
        )
        step3b_update.next_to(step3b_gradient, DOWN, buff=0.2)

        self.play(Write(step3b_gradient), run_time=1)
        self.wait(0.5)
        self.play(Write(step3b_update), run_time=1.5)
        
        # Blink effect on b1 label
        self.play(
            self.bias_labels[0].animate.set_color(YELLOW).scale(1.3),
            run_time=0.3
        )
        self.play(
            self.bias_labels[0].animate.set_color(PURPLE).scale(1/1.3),
            run_time=0.3
        )
        self.wait(1)

        # Final backwards arrow to input
        back_arrow3 = Arrow(
            self.neurons[1].get_left() + LEFT * 0.1,
            self.neurons[0].get_right() + RIGHT * 0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(Create(back_arrow3), run_time=1.5)
        self.wait(2)
        
    def show_complex_gradient_architecture(self):
        """Show a more complex network with gradient vectors in column notation"""
        
        # Title
        title = Text("Gradientes em Rede Neural Completa", font_size=36, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(1)
        
        # Build network: 3 inputs -> 4 hidden -> 2 output
        # Input layer (3 neurons)
        input_layer = VGroup()
        input_positions = [UP * 1.5, ORIGIN, DOWN * 1.5]
        for i, pos in enumerate(input_positions):
            neuron = Circle(radius=0.25, color=BLUE, fill_opacity=0.3)
            neuron.shift(LEFT * 5 + pos)
            label = MathTex(f"x_{i+1}", font_size=20).move_to(neuron)
            input_layer.add(VGroup(neuron, label))
        
        # Hidden layer (4 neurons)
        hidden_layer = VGroup()
        hidden_positions = [UP * 2, UP * 0.7, DOWN * 0.7, DOWN * 2]
        for i, pos in enumerate(hidden_positions):
            neuron = Circle(radius=0.25, color=YELLOW, fill_opacity=0.2)
            neuron.shift(LEFT * 1.5 + pos)
            label = MathTex(f"h_{i+1}", font_size=20).move_to(neuron)
            hidden_layer.add(VGroup(neuron, label))
        
        # Output layer (2 neurons)
        output_layer = VGroup()
        output_positions = [UP * 0.8, DOWN * 0.8]
        for i, pos in enumerate(output_positions):
            neuron = Circle(radius=0.25, color=GREEN, fill_opacity=0.3)
            neuron.shift(RIGHT * 2.5 + pos)
            label = MathTex(f"y_{i+1}", font_size=20).move_to(neuron)
            output_layer.add(VGroup(neuron, label))
        
        # Create connections (simplified - not all)
        edges_input_hidden = VGroup()
        for inp in input_layer:
            for hid in hidden_layer:
                edge = Line(
                    inp[0].get_center(), 
                    hid[0].get_center(), 
                    color=GRAY, 
                    stroke_width=1,
                    stroke_opacity=0.3
                )
                edges_input_hidden.add(edge)
        
        edges_hidden_output = VGroup()
        for hid in hidden_layer:
            for out in output_layer:
                edge = Line(
                    hid[0].get_center(), 
                    out[0].get_center(), 
                    color=GRAY, 
                    stroke_width=1,
                    stroke_opacity=0.3
                )
                edges_hidden_output.add(edge)
        
        # Draw network
        self.play(
            Create(edges_input_hidden),
            Create(edges_hidden_output),
            run_time=1
        )
        self.play(
            LaggedStart(*[Create(group) for group in input_layer], lag_ratio=0.1),
            LaggedStart(*[Create(group) for group in hidden_layer], lag_ratio=0.1),
            LaggedStart(*[Create(group) for group in output_layer], lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(1)
        
        # Gradient vectors in column notation
        # Layer 1 gradients (W1: 4x3 matrix, b1: 4x1)
        grad_w1_title = Text("Camada 1:", font_size=24, color=YELLOW).shift(LEFT * 5 + DOWN * 2.5)
        
        grad_w1 = MathTex(
            r"\nabla W_1 = \begin{bmatrix} \frac{\partial L}{\partial w_{11}} \\ \frac{\partial L}{\partial w_{21}} \\ \vdots \\ \frac{\partial L}{\partial w_{43}} \end{bmatrix}",
            font_size=22,
            color=ORANGE
        ).next_to(grad_w1_title, RIGHT, buff=0.3)
        
        grad_b1 = MathTex(
            r"\nabla b_1 = \begin{bmatrix} \frac{\partial L}{\partial b_1} \\ \frac{\partial L}{\partial b_2} \\ \frac{\partial L}{\partial b_3} \\ \frac{\partial L}{\partial b_4} \end{bmatrix}",
            font_size=22,
            color=PURPLE
        ).next_to(grad_w1, RIGHT, buff=0.5)
        
        # Layer 2 gradients (W2: 2x4 matrix, b2: 2x1)
        grad_w2_title = Text("Camada 2:", font_size=24, color=GREEN).shift(RIGHT * 0.5 + DOWN * 2.5)
        
        grad_w2 = MathTex(
            r"\nabla W_2 = \begin{bmatrix} \frac{\partial L}{\partial w_{11}} \\ \vdots \\ \frac{\partial L}{\partial w_{24}} \end{bmatrix}",
            font_size=22,
            color=ORANGE
        ).next_to(grad_w2_title, RIGHT, buff=0.3)
        
        grad_b2 = MathTex(
            r"\nabla b_2 = \begin{bmatrix} \frac{\partial L}{\partial b_1} \\ \frac{\partial L}{\partial b_2} \end{bmatrix}",
            font_size=22,
            color=PURPLE
        ).next_to(grad_w2, RIGHT, buff=0.5)
        
        # Show gradients
        self.play(Write(grad_w1_title), run_time=0.8)
        self.wait(0.3)
        self.play(Write(grad_w1), run_time=1.5)
        self.wait(0.5)
        self.play(Write(grad_b1), run_time=1.5)
        self.wait(1)
        
        self.play(Write(grad_w2_title), run_time=0.8)
        self.wait(0.3)
        self.play(Write(grad_w2), run_time=1.5)
        self.wait(0.5)
        self.play(Write(grad_b2), run_time=1.5)
        self.wait(2)
        
        # Summary text
        summary = Text(
            "Cada peso e bias tem seu gradiente calculado",
            font_size=26,
            color=YELLOW,
            slant=ITALIC
        )
        summary.to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(summary, shift=UP * 0.3))
        self.wait(2)
