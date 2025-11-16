from manim import *
import numpy as np


class SimplifiedPath(Scene):
    """Single path through simplified neural network with glossary"""

    def construct(self):
        # First show glossary
        self.show_glossary()
        self.wait(3)
        
        # Clear and show the path
        title = Text("Caminho através da Rede Neural", font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        self.show_single_path()
        self.wait(3)

    def show_glossary(self):
        """Show glossary of terms"""
        glossary_title = Text("Glossário:", font_size=44, weight=BOLD, color=YELLOW)
        glossary_title.to_edge(UP, buff=0.5)
        
        terms = VGroup(
            VGroup(
                MathTex("x_1, x_2, ...", font_size=28, color=BLUE),
                Text("= Features de entrada (dados)", font_size=22),
            ).arrange(RIGHT, buff=0.3),
            
            VGroup(
                MathTex("w", font_size=28, color=RED),
                Text("= Pesos (weights) - aprendidos pelo treinamento", font_size=22),
            ).arrange(RIGHT, buff=0.3),
            
            VGroup(
                MathTex("b", font_size=28, color=PURPLE),
                Text("= Bias - termo de ajuste", font_size=22),
            ).arrange(RIGHT, buff=0.3),
            
            VGroup(
                MathTex(r"\sigma", font_size=28, color=GREEN),
                Text("= Função de ativação (ex: sigmoid, ReLU)", font_size=22),
            ).arrange(RIGHT, buff=0.3),
            
            VGroup(
                MathTex(r"\mathcal{L}", font_size=28, color=RED),
                Text("= Loss function - mede o erro da predição", font_size=22),
            ).arrange(RIGHT, buff=0.3),
            
            VGroup(
                MathTex(r"\hat{y}", font_size=28, color=GREEN),
                Text("= Predição da rede", font_size=22),
            ).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        terms.next_to(glossary_title, DOWN, buff=0.6)
        
        self.play(Write(glossary_title), run_time=1.5)
        self.wait(1)
        
        for term in terms:
            self.play(Write(term), run_time=1.2)
            self.wait(0.8)
        
        self.wait(2)
        self.play(FadeOut(glossary_title), FadeOut(terms), run_time=1.5)
        self.wait(1)

    def show_single_path(self):
        """Show x1 → h (with bias) → σ → Loss → ŷ"""
        
        # Input
        x1 = Circle(radius=0.3, color=BLUE, fill_opacity=0.3).shift(LEFT * 5.5)
        x1_label = MathTex("x_1", font_size=28, color=BLUE).move_to(x1)
        
        # Weight
        w_line = Line(x1.get_right(), LEFT * 3.2, color=GRAY, stroke_width=3)
        w_label = MathTex("w", font_size=24, color=RED).next_to(w_line, UP, buff=0.2)
        
        # Hidden neuron (circle)
        h = Circle(radius=0.4, color=YELLOW, fill_opacity=0.2, stroke_width=3).shift(LEFT * 2.5)
        h_label = MathTex("h", font_size=28, color=YELLOW).move_to(h)
        
        # Bias above neuron
        b = Circle(radius=0.25, color=PURPLE, fill_opacity=0.4).move_to(h.get_top() + UP * 0.9)
        b_label = MathTex("b", font_size=24, color=PURPLE).move_to(b)
        b_arrow = Arrow(b.get_bottom(), h.get_top(), color=PURPLE, stroke_width=2, buff=0.05)
        
        # Activation
        h_to_sigma = Line(h.get_right(), RIGHT * 0.2, color=GRAY, stroke_width=3)
        sigma = Circle(radius=0.35, color=GREEN, stroke_width=3).shift(RIGHT * 0.8)
        sigma_label = MathTex(r"\sigma", font_size=28, color=GREEN).move_to(sigma)
        
        # Loss function
        sigma_to_loss = Line(sigma.get_right(), RIGHT * 2.8, color=GRAY, stroke_width=3)
        loss_box = Rectangle(height=0.8, width=1.2, color=RED, stroke_width=3).shift(RIGHT * 3.5)
        loss_label = MathTex("L", font_size=32, color=RED).move_to(loss_box)
        
        # Output ŷ
        loss_to_yhat = Arrow(loss_box.get_right(), RIGHT * 5.5, color=GREEN, stroke_width=3)
        yhat = MathTex(r"\hat{y}", font_size=32, color=GREEN).shift(RIGHT * 6)
        
        # Animate construction
        self.play(Create(x1), Write(x1_label), run_time=1.2)
        self.wait(1)
        
        self.play(Create(w_line), Write(w_label), run_time=1.5)
        self.wait(0.8)
        
        self.play(Create(h), Write(h_label), run_time=1.5)
        self.wait(1)
        
        self.play(Create(b), Write(b_label), run_time=1.2)
        self.wait(0.5)
        self.play(Create(b_arrow), run_time=1)
        self.wait(1.5)
        
        self.play(Create(h_to_sigma), run_time=1)
        self.wait(0.5)
        self.play(Create(sigma), Write(sigma_label), run_time=1.5)
        self.wait(1.5)
        
        self.play(Create(sigma_to_loss), run_time=1)
        self.wait(0.5)
        self.play(Create(loss_box), Write(loss_label), run_time=1.5)
        self.wait(1.5)
        
        self.play(Create(loss_to_yhat), run_time=1.2)
        self.wait(0.5)
        self.play(Write(yhat), run_time=1.2)
        self.wait(2)
        
        # Show the formula
        self.show_path_formula()

    def show_path_formula(self):
        """Show the complete formula for this path"""
        
        formula_title = Text("Fórmula completa:", font_size=28, weight=BOLD, color=YELLOW)
        formula_title.to_edge(DOWN, buff=2.5)
        
        formula = MathTex(
            r"\hat{y} = L(\sigma(w \cdot x_1 + b))",
            font_size=40,
            color=GREEN,
        ).next_to(formula_title, DOWN, buff=0.5)
        
        box = SurroundingRectangle(formula, color=YELLOW, buff=0.3)
        
        self.play(Write(formula_title), run_time=1.5)
        self.wait(0.8)
        self.play(Write(formula), run_time=2.5)
        self.wait(1.5)
        self.play(Create(box), run_time=1.2)
        self.wait(2.5)


class NeuralNetworkArchitecture(Scene):
    """Full neural network architecture - clean visualization without formulas"""

    def construct(self):
        title = Text("Arquitetura da Rede Neural Completa", font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Build horizontal network with 2 hidden layers
        self.build_horizontal_network()
        self.wait(3)

    def build_horizontal_network(self):
        """Build network with horizontal layout - 2 hidden layers and bias nodes"""

        # Input layer (left side)
        inputs = VGroup(
            Circle(radius=0.22, color=BLUE, fill_opacity=0.3).shift(
                LEFT * 5.5 + UP * 0.6
            ),
            Circle(radius=0.22, color=BLUE, fill_opacity=0.3).shift(
                LEFT * 5.5 + DOWN * 0.6
            ),
        )
        input_labels = VGroup(
            MathTex("x_1", font_size=18).move_to(inputs[0]),
            MathTex("x_2", font_size=18).move_to(inputs[1]),
        )
        input_text = Text("Input", font_size=20).next_to(
            inputs, DOWN, buff=0.5
        )

        # Hidden layer 1 (center-left)
        hidden1 = VGroup(
            Circle(radius=0.22, color=YELLOW, fill_opacity=0.2).shift(
                LEFT * 3 + UP * 0.9
            ),
            Circle(radius=0.22, color=YELLOW, fill_opacity=0.2).shift(
                LEFT * 3
            ),
            Circle(radius=0.22, color=YELLOW, fill_opacity=0.2).shift(
                LEFT * 3 + DOWN * 0.9
            ),
        )
        hidden1_labels = VGroup(
            MathTex("h_1", font_size=16, color=YELLOW).move_to(hidden1[0]),
            MathTex("h_2", font_size=16, color=YELLOW).move_to(hidden1[1]),
            MathTex("h_3", font_size=16, color=YELLOW).move_to(hidden1[2]),
        )
        hidden1_text = Text("Hidden 1", font_size=20).next_to(
            hidden1, DOWN, buff=0.5
        )

        # Hidden layer 2 (center)
        hidden2 = VGroup(
            Circle(radius=0.22, color=ORANGE, fill_opacity=0.2).shift(
                LEFT * 0.5 + UP * 0.9
            ),
            Circle(radius=0.22, color=ORANGE, fill_opacity=0.2).shift(
                LEFT * 0.5
            ),
            Circle(radius=0.22, color=ORANGE, fill_opacity=0.2).shift(
                LEFT * 0.5 + DOWN * 0.9
            ),
        )
        hidden2_labels = VGroup(
            MathTex("h_4", font_size=16, color=ORANGE).move_to(hidden2[0]),
            MathTex("h_5", font_size=16, color=ORANGE).move_to(hidden2[1]),
            MathTex("h_6", font_size=16, color=ORANGE).move_to(hidden2[2]),
        )
        hidden2_text = Text("Hidden 2", font_size=20).next_to(
            hidden2, DOWN, buff=0.5
        )

        # Output layer (center-right)
        output = Circle(radius=0.22, color=GREEN, fill_opacity=0.3).shift(
            RIGHT * 2.2
        )
        output_text = Text("Output", font_size=20).next_to(
            output, DOWN, buff=0.5
        )

        # Loss (right side)
        loss_box = Rectangle(
            height=0.6, width=1, color=RED, stroke_width=2
        ).shift(RIGHT * 4.5)
        loss_label = MathTex("L", font_size=28, color=RED).move_to(loss_box)
        loss_text = Text("Loss", font_size=20).next_to(
            loss_box, DOWN, buff=0.35
        )

        # ŷ appears AFTER loss (rightmost)
        y_hat_label = MathTex(r"\hat{y}", font_size=28, color=GREEN).shift(
            RIGHT * 6.2
        )

        # Offsets para os biases - posicionados à direita (acima e à direita)
        bias_diagonal_offset = RIGHT * 0.8 + UP * 0.6

        # Biases para hidden1 - um por neurônio posicionado diagonalmente
        bias1_nodes = VGroup(
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden1[0].get_center() + bias_diagonal_offset
            ),
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden1[1].get_center() + bias_diagonal_offset
            ),
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden1[2].get_center() + bias_diagonal_offset
            ),
        )
        bias1_labels = VGroup(
            MathTex("b_1", font_size=14, color=PURPLE).move_to(bias1_nodes[0]),
            MathTex("b_2", font_size=14, color=PURPLE).move_to(bias1_nodes[1]),
            MathTex("b_3", font_size=14, color=PURPLE).move_to(bias1_nodes[2]),
        )

        # Biases para hidden2 - um por neurônio posicionado diagonalmente
        bias2_nodes = VGroup(
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden2[0].get_center() + bias_diagonal_offset
            ),
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden2[1].get_center() + bias_diagonal_offset
            ),
            Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
                hidden2[2].get_center() + bias_diagonal_offset
            ),
        )
        bias2_labels = VGroup(
            MathTex("b_4", font_size=14, color=PURPLE).move_to(bias2_nodes[0]),
            MathTex("b_5", font_size=14, color=PURPLE).move_to(bias2_nodes[1]),
            MathTex("b_6", font_size=14, color=PURPLE).move_to(bias2_nodes[2]),
        )

        # Bias do neurônio de saída - posicionado diagonalmente também
        bias3 = Circle(radius=0.15, color=PURPLE, fill_opacity=0.4).move_to(
            output.get_center() + bias_diagonal_offset
        )
        bias3_label = MathTex("b_7", font_size=14, color=PURPLE).move_to(bias3)

        # Connections input -> hidden1
        input_hidden1_lines = VGroup()
        for inp in inputs:
            for hid in hidden1:
                line = Line(
                    inp.get_right(),
                    hid.get_left(),
                    color=GRAY,
                    stroke_width=1.2,
                    stroke_opacity=0.5,
                )
                input_hidden1_lines.add(line)

        # Bias1 -> hidden1 - arrow from bias (right) to neuron (left)
        bias1_lines = VGroup()
        for b_node, hid in zip(bias1_nodes, hidden1):
            arrow = Arrow(
                b_node.get_left(),
                hid.get_right(),
                color=PURPLE,
                stroke_width=2,
                stroke_opacity=0.6,
                buff=0.15,
            )
            bias1_lines.add(arrow)
        
        # Weight label (just one as example)
        weight_label = MathTex("w", font_size=16, color=RED).move_to(
            input_hidden1_lines[1].get_center() + UP * 0.25
        )

        # Connections hidden1 -> hidden2
        hidden1_hidden2_lines = VGroup()
        for hid1 in hidden1:
            for hid2 in hidden2:
                line = Line(
                    hid1.get_right(),
                    hid2.get_left(),
                    color=GRAY,
                    stroke_width=1.2,
                    stroke_opacity=0.5,
                )
                hidden1_hidden2_lines.add(line)

        # Bias2 -> hidden2 - arrow from bias (right) to neuron (left)
        bias2_lines = VGroup()
        for b_node, hid in zip(bias2_nodes, hidden2):
            arrow = Arrow(
                b_node.get_left(),
                hid.get_right(),
                color=PURPLE,
                stroke_width=2,
                stroke_opacity=0.6,
                buff=0.15,
            )
            bias2_lines.add(arrow)

        # Connections hidden2 -> output
        hidden2_output_lines = VGroup()
        for hid in hidden2:
            line = Line(
                hid.get_right(),
                output.get_left(),
                color=GRAY,
                stroke_width=1.2,
                stroke_opacity=0.5,
            )
            hidden2_output_lines.add(line)

        # Bias3 -> output - arrow from bias (right) to neuron (left)
        bias3_line = Arrow(
            bias3.get_left(),
            output.get_right(),
            color=PURPLE,
            stroke_width=2,
            stroke_opacity=0.6,
            buff=0.15,
        )

        # Output -> Loss -> ŷ
        output_loss_line = Line(
            output.get_right(), loss_box.get_left(), color=RED, stroke_width=2
        )

        loss_yhat_arrow = Arrow(
            loss_box.get_right(),
            y_hat_label.get_left() + LEFT * 0.1,
            color=GREEN,
            stroke_width=3,
        )

        # Animações
        self.play(
            LaggedStart(
                *[Create(inp) for inp in inputs],
                lag_ratio=0.3,
            ),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[Write(label) for label in input_labels], lag_ratio=0.2),
            run_time=1,
        )
        self.play(Write(input_text), run_time=0.8)
        self.wait(0.8)

        # Bias e conexões da primeira hidden
        self.play(
            Create(bias1_nodes),
            run_time=1,
        )
        self.play(
            LaggedStart(*[Write(label) for label in bias1_labels], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.5)

        self.play(
            Create(input_hidden1_lines),
            Create(bias1_lines),
            run_time=1.5,
        )
        self.play(Write(weight_label), run_time=0.8)
        self.wait(0.5)

        self.play(
            LaggedStart(*[Create(h) for h in hidden1], lag_ratio=0.2),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[Write(label) for label in hidden1_labels], lag_ratio=0.2),
            run_time=1,
        )
        self.play(Write(hidden1_text), run_time=0.8)
        self.wait(0.8)

        # Bias e conexões da segunda hidden
        self.play(
            Create(bias2_nodes),
            run_time=1,
        )
        self.play(
            LaggedStart(*[Write(label) for label in bias2_labels], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.5)

        self.play(
            Create(hidden1_hidden2_lines),
            Create(bias2_lines),
            run_time=1.5,
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*[Create(h) for h in hidden2], lag_ratio=0.2),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[Write(label) for label in hidden2_labels], lag_ratio=0.2),
            run_time=1,
        )
        self.play(Write(hidden2_text), run_time=0.8)
        self.wait(0.8)

        # Bias e conexões do output
        self.play(Create(bias3), run_time=1)
        self.play(Write(bias3_label), run_time=0.8)
        self.wait(0.5)

        self.play(
            Create(hidden2_output_lines),
            Create(bias3_line),
            run_time=1.5,
        )
        self.wait(0.5)

        self.play(Create(output), run_time=1.2)
        self.play(Write(output_text), run_time=0.8)
        self.wait(0.8)

        self.play(Create(output_loss_line), run_time=1.2)
        self.wait(0.5)

        self.play(Create(loss_box), Write(loss_label), run_time=1.2)
        self.play(Write(loss_text), run_time=0.8)
        self.wait(0.8)

        # ŷ depois da loss
        self.play(Create(loss_yhat_arrow), run_time=1.5)
        self.wait(0.5)
        self.play(Write(y_hat_label), run_time=1.2)
        self.wait(1.5)

