from manim import *
import numpy as np

# Configuração para aspecto widescreen 16:9
config.frame_width = 16
config.frame_height = 9
config.pixel_width = 1920
config.pixel_height = 1080

# ==============================================================================
# ESTRUTURA:
# 1. NeuralNetworkAsFunction - Rede Neural como Função
# 2. ActivationFunctions - Exemplos de Funções de Ativação
# 3. LossFunctions - Exemplos de Funções de Perda
#
# Renderizar tudo:
#   manim -pqh neural_network_intro.py
# ==============================================================================


class NeuralNetworkAsFunction(Scene):
    """Progressão: Função → Exemplo → Arquitetura → Composição"""

    def construct(self):
        # 1. Rede Neural como Função
        self.show_as_function()
        self.wait(2)

        # 2. Exemplo de Regressão
        self.show_regression_example()
        self.wait(2)

        # 3. Arquitetura Visual Específica
        self.show_architecture()
        self.wait(2)

        # 4. Composição Matemática daquela rede
        self.show_composition()
        self.wait(3)

    def show_as_function(self):
        """Mostra rede neural como função"""
        
        # Subtítulo
        subtitle = Text("Rede Neural como Função", font_size=36, color=WHITE)
        subtitle.to_edge(UP, buff=0.3)
        self.play(Write(subtitle), run_time=1.2)
        self.wait(0.8)
        
        # Caixa da função
        function_box = RoundedRectangle(
            height=2.5,
            width=4,
            corner_radius=0.2,
            stroke_color=BLUE,
            stroke_width=4,
            fill_opacity=0.1,
            fill_color=BLUE
        )
        
        nn_text = Text("Rede\nNeural", font_size=36, weight=BOLD)
        nn_text.move_to(function_box)
        
        # Entrada
        input_label = MathTex(r"\mathbf{x}", font_size=48, color=GREEN)
        input_label.next_to(function_box, LEFT, buff=1.5)
        
        arrow_in = Arrow(
            input_label.get_right(),
            function_box.get_left(),
            color=GREEN,
            stroke_width=6,
            buff=0.2
        )
        
        # Saída
        output_label = MathTex(r"\mathbf{y}", font_size=48, color=ORANGE)
        output_label.next_to(function_box, RIGHT, buff=1.5)
        
        arrow_out = Arrow(
            function_box.get_right(),
            output_label.get_left(),
            color=ORANGE,
            stroke_width=6,
            buff=0.2
        )
        
        # Função matemática
        math_function = MathTex(
            r"f_{\theta}(\mathbf{x}) = \mathbf{y}",
            font_size=44,
            color=YELLOW
        )
        math_function.next_to(function_box, DOWN, buff=1.2)
        
        # Animar
        self.play(Create(function_box), Write(nn_text), run_time=1.5)
        self.wait(0.5)
        
        self.play(
            Write(input_label),
            Create(arrow_in),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Write(output_label),
            Create(arrow_out),
            run_time=1
        )
        self.wait(0.8)
        
        self.play(Write(math_function), run_time=1.5)
        self.wait(2)
        
        # Guardar
        self.function_group = VGroup(
            subtitle, function_box, nn_text, input_label, arrow_in,
            output_label, arrow_out, math_function
        )

    def show_regression_example(self):
        """Mostra exemplo de regressão com 5 entradas"""
        
        self.play(FadeOut(self.function_group), run_time=0.8)
        
        example_title = Text("Exemplo: Prever Preço de Casa", font_size=36, color=WHITE)
        example_title.to_edge(UP, buff=0.3)
        
        self.play(Write(example_title), run_time=1.2)
        
        # 5 entradas
        inputs = VGroup(
            VGroup(
                Text("Área (m²)", font_size=18, color=GREEN),
                MathTex(r"x_1 = 120", font_size=20)
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Text("Quartos", font_size=18, color=GREEN),
                MathTex(r"x_2 = 3", font_size=20)
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Text("Banheiros", font_size=18, color=GREEN),
                MathTex(r"x_3 = 2", font_size=20)
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Text("Idade (anos)", font_size=18, color=GREEN),
                MathTex(r"x_4 = 5", font_size=20)
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Text("Dist. Centro (km)", font_size=18, color=GREEN),
                MathTex(r"x_5 = 8", font_size=20)
            ).arrange(DOWN, buff=0.15),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        inputs.shift(LEFT * 4 + DOWN * 0.3)
        
        # Vetor de entrada
        input_vector = MathTex(
            r"\mathbf{x} = \begin{bmatrix} 120 \\ 3 \\ 2 \\ 5 \\ 8 \end{bmatrix}",
            font_size=28,
            color=GREEN
        )
        input_vector.next_to(inputs, RIGHT, buff=1)
        
        # Rede
        nn_box = RoundedRectangle(
            height=2.2,
            width=2.2,
            corner_radius=0.2,
            stroke_color=BLUE,
            stroke_width=4,
            fill_opacity=0.1,
            fill_color=BLUE
        )
        nn_box.next_to(input_vector, RIGHT, buff=0.8)
        
        nn_label = MathTex(r"f_{\theta}", font_size=36)
        nn_label.move_to(nn_box)
        
        arrow1 = Arrow(input_vector.get_right(), nn_box.get_left(), 
                      color=WHITE, stroke_width=4, buff=0.2)
        
        # Saída
        output = VGroup(
            Text("Preço", font_size=20, color=ORANGE),
            MathTex(r"y = \text{R\$} 450k", font_size=24, color=ORANGE)
        ).arrange(DOWN, buff=0.15)
        output.next_to(nn_box, RIGHT, buff=0.8)
        
        arrow2 = Arrow(nn_box.get_right(), output.get_left(),
                      color=WHITE, stroke_width=4, buff=0.2)
        
        # Animar
        self.play(Write(inputs), run_time=2)
        self.wait(0.8)
        
        self.play(Write(input_vector), run_time=1.2)
        self.wait(0.5)
        
        self.play(
            Create(arrow1),
            Create(nn_box),
            Write(nn_label),
            run_time=1.2
        )
        self.wait(0.5)
        
        self.play(
            Create(arrow2),
            Write(output),
            run_time=1.2
        )
        self.wait(2)
        
        # Guardar
        self.example_group = VGroup(
            example_title, inputs, input_vector, nn_box, nn_label,
            arrow1, arrow2, output
        )

    def show_architecture(self):
        """Mostra arquitetura visual: 5→3→3→1"""
        
        self.play(FadeOut(self.example_group), run_time=0.8)
        
        arch_title = Text("Arquitetura da Rede", font_size=36, color=WHITE)
        arch_title.to_edge(UP, buff=0.3)
        
        self.play(Write(arch_title), run_time=1.2)
        self.wait(0.5)
        
        # Criar camadas
        # Entrada: 5 neurônios
        input_layer = VGroup(*[
            Circle(radius=0.2, color=GREEN, fill_opacity=0.3)
            for _ in range(5)
        ]).arrange(DOWN, buff=0.4)
        input_layer.shift(LEFT * 5.5 + DOWN * 0.2)
        
        input_labels = VGroup(
            MathTex(r"x_1", font_size=18).next_to(input_layer[0], LEFT, buff=0.15),
            MathTex(r"x_2", font_size=18).next_to(input_layer[1], LEFT, buff=0.15),
            MathTex(r"x_3", font_size=18).next_to(input_layer[2], LEFT, buff=0.15),
            MathTex(r"x_4", font_size=18).next_to(input_layer[3], LEFT, buff=0.15),
            MathTex(r"x_5", font_size=18).next_to(input_layer[4], LEFT, buff=0.15)
        )
        
        # Hidden Layer 1: 3 neurônios
        hidden1 = VGroup(*[
            Circle(radius=0.2, color=BLUE, fill_opacity=0.3)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.5)
        hidden1.shift(LEFT * 3 + DOWN * 0.2)
        
        # Hidden Layer 2: 3 neurônios
        hidden2 = VGroup(*[
            Circle(radius=0.2, color=BLUE, fill_opacity=0.3)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.5)
        hidden2.shift(LEFT * 0.5 + DOWN * 0.2)
        
        # Output: 1 neurônio
        output_layer = VGroup(
            Circle(radius=0.2, color=ORANGE, fill_opacity=0.3)
        )
        output_layer.shift(RIGHT * 2 + DOWN * 0.2)
        
        output_label = MathTex(r"y", font_size=20).next_to(output_layer, RIGHT, buff=0.2)
        
        # Bias nodes - posicionados diagonalmente acima à direita de cada camada
        bias_diagonal = RIGHT * 0.6 + UP * 0.5
        
        # Bias para hidden1
        bias1_nodes = VGroup(*[
            Circle(radius=0.12, color=PURPLE, fill_opacity=0.5)
            for _ in range(3)
        ])
        for i, (b_node, h_node) in enumerate(zip(bias1_nodes, hidden1)):
            b_node.move_to(h_node.get_center() + bias_diagonal)
        
        bias1_labels = VGroup(*[
            MathTex(f"b_{i+1}", font_size=14, color=PURPLE).move_to(bias1_nodes[i])
            for i in range(3)
        ])
        
        # Bias para hidden2
        bias2_nodes = VGroup(*[
            Circle(radius=0.12, color=PURPLE, fill_opacity=0.5)
            for _ in range(3)
        ])
        for i, (b_node, h_node) in enumerate(zip(bias2_nodes, hidden2)):
            b_node.move_to(h_node.get_center() + bias_diagonal)
        
        bias2_labels = VGroup(*[
            MathTex(f"b_{i+4}", font_size=14, color=PURPLE).move_to(bias2_nodes[i])
            for i in range(3)
        ])
        
        # Bias para output
        bias3_node = Circle(radius=0.12, color=PURPLE, fill_opacity=0.5)
        bias3_node.move_to(output_layer[0].get_center() + bias_diagonal)
        bias3_label = MathTex(r"b_7", font_size=14, color=PURPLE).move_to(bias3_node)
        
        # Conexões com pesos
        connections = VGroup()
        
        # Input → Hidden1 (mostrar alguns pesos)
        for i, n1 in enumerate(input_layer):
            for j, n2 in enumerate(hidden1):
                line = Line(n1.get_center(), n2.get_center(),
                          stroke_width=1, color=GRAY, stroke_opacity=0.4)
                connections.add(line)
                
                # Mostrar um label de peso como exemplo
                if i == 0 and j == 0:
                    w_label = MathTex(r"w", font_size=14, color=RED)
                    w_label.move_to(line.get_center() + UP * 0.2)
                    connections.add(w_label)
        
        # Bias1 → Hidden1 (arrows diagonais)
        bias1_connections = VGroup()
        for b_node, h_node in zip(bias1_nodes, hidden1):
            arrow = Arrow(
                b_node.get_left(),
                h_node.get_right(),
                color=PURPLE,
                stroke_width=1.5,
                buff=0.12,
                max_tip_length_to_length_ratio=0.15
            )
            bias1_connections.add(arrow)
        
        # Sigma symbols após hidden1
        sigma1_symbols = VGroup()
        for i, h_node in enumerate(hidden1):
            sigma_pos = h_node.get_right() + RIGHT * 0.35
            sigma = MathTex(r"\sigma", font_size=16, color=GREEN)
            sigma.move_to(sigma_pos)
            sigma1_symbols.add(sigma)
        
        # Hidden1 → Hidden2
        hidden1_hidden2_lines = VGroup()
        for n1 in hidden1:
            for n2 in hidden2:
                # Linha começa após o sigma
                start_pos = n1.get_right() + RIGHT * 0.7
                line = Line(start_pos, n2.get_center(),
                          stroke_width=1, color=GRAY, stroke_opacity=0.4)
                hidden1_hidden2_lines.add(line)
        
        # Bias2 → Hidden2 (arrows diagonais)
        bias2_connections = VGroup()
        for b_node, h_node in zip(bias2_nodes, hidden2):
            arrow = Arrow(
                b_node.get_left(),
                h_node.get_right(),
                color=PURPLE,
                stroke_width=1.5,
                buff=0.12,
                max_tip_length_to_length_ratio=0.15
            )
            bias2_connections.add(arrow)
        
        # Sigma symbols após hidden2
        sigma2_symbols = VGroup()
        for i, h_node in enumerate(hidden2):
            sigma_pos = h_node.get_right() + RIGHT * 0.35
            sigma = MathTex(r"\sigma", font_size=16, color=GREEN)
            sigma.move_to(sigma_pos)
            sigma2_symbols.add(sigma)
        
        # Hidden2 → Output
        hidden2_output_lines = VGroup()
        for n1 in hidden2:
            # Linha começa após o sigma
            start_pos = n1.get_right() + RIGHT * 0.7
            line = Line(start_pos, output_layer[0].get_center(),
                      stroke_width=1, color=GRAY, stroke_opacity=0.4)
            hidden2_output_lines.add(line)
        
        # Bias3 → Output (arrow diagonal)
        bias3_connection = Arrow(
            bias3_node.get_left(),
            output_layer[0].get_right(),
            color=PURPLE,
            stroke_width=1.5,
            buff=0.12,
            max_tip_length_to_length_ratio=0.15
        )
        
        # Labels das camadas
        layer_labels = VGroup(
            Text("Entrada (5)", font_size=16, color=GREEN).next_to(input_layer, DOWN, buff=0.4),
            Text("Hidden 1 (3)", font_size=16, color=BLUE).next_to(hidden1, DOWN, buff=0.4),
            Text("Hidden 2 (3)", font_size=16, color=BLUE).next_to(hidden2, DOWN, buff=0.4),
            Text("Saída (1)", font_size=16, color=ORANGE).next_to(output_layer, DOWN, buff=0.4)
        )
        
        # Notações - reorganizadas em uma única linha
        weight_note = MathTex(
            r"w = \text{peso}",
            font_size=18,
            color=RED
        )
        weight_note.to_edge(DOWN, buff=0.8).shift(LEFT * 4)
        
        bias_note = MathTex(
            r"b = \text{bias}",
            font_size=18,
            color=PURPLE
        )
        bias_note.next_to(weight_note, RIGHT, buff=1)
        
        activation_note = MathTex(
            r"\sigma = \text{ativação}",
            font_size=18,
            color=GREEN
        )
        activation_note.next_to(bias_note, RIGHT, buff=1)
        
        # Animar - Input layer
        self.play(
            LaggedStart(
                *[Create(n) for n in input_layer],
                lag_ratio=0.1
            ),
            run_time=1.5
        )
        self.play(Write(input_labels), run_time=1)
        self.play(Write(layer_labels[0]), run_time=0.8)
        self.wait(0.5)
        
        # Conexões e Hidden1
        self.play(Create(connections), run_time=1.5)
        self.wait(0.5)
        
        self.play(
            Create(bias1_nodes),
            Write(bias1_labels),
            run_time=1
        )
        self.play(Create(bias1_connections), run_time=1)
        self.wait(0.5)
        
        self.play(
            LaggedStart(
                *[Create(n) for n in hidden1],
                lag_ratio=0.1
            ),
            run_time=1.2
        )
        self.play(Write(layer_labels[1]), run_time=0.8)
        self.wait(0.5)
        
        # Sigma após hidden1
        self.play(Write(sigma1_symbols), run_time=1)
        self.wait(0.5)
        
        # Conexões para Hidden2
        self.play(Create(hidden1_hidden2_lines), run_time=1.5)
        self.wait(0.5)
        
        self.play(
            Create(bias2_nodes),
            Write(bias2_labels),
            run_time=1
        )
        self.play(Create(bias2_connections), run_time=1)
        self.wait(0.5)
        
        self.play(
            LaggedStart(
                *[Create(n) for n in hidden2],
                lag_ratio=0.1
            ),
            run_time=1.2
        )
        self.play(Write(layer_labels[2]), run_time=0.8)
        self.wait(0.5)
        
        # Sigma após hidden2
        self.play(Write(sigma2_symbols), run_time=1)
        self.wait(0.5)
        
        # Conexões para Output
        self.play(Create(hidden2_output_lines), run_time=1.5)
        self.wait(0.5)
        
        self.play(
            Create(bias3_node),
            Write(bias3_label),
            run_time=1
        )
        self.play(Create(bias3_connection), run_time=1)
        self.wait(0.5)
        
        self.play(Create(output_layer), run_time=1.2)
        self.play(
            Write(output_label),
            Write(layer_labels[3]),
            run_time=1
        )
        self.wait(1)
        
        # Loss function box
        loss_box = Rectangle(
            height=0.8,
            width=1.2,
            color=RED,
            stroke_width=3,
            fill_opacity=0.1,
            fill_color=RED
        )
        loss_box.next_to(output_layer, RIGHT, buff=1)
        
        loss_label = MathTex(r"\mathcal{L}", font_size=32, color=RED)
        loss_label.move_to(loss_box)
        
        loss_text = Text("Loss", font_size=16, color=RED)
        loss_text.next_to(loss_box, DOWN, buff=0.3)
        
        output_to_loss = Line(
            output_layer[0].get_right(),
            loss_box.get_left(),
            color=RED,
            stroke_width=2
        )
        
        # Animar Loss
        self.play(Create(output_to_loss), run_time=1)
        self.wait(0.5)
        self.play(
            Create(loss_box),
            Write(loss_label),
            run_time=1.2
        )
        self.play(Write(loss_text), run_time=0.8)
        self.wait(1.5)
        
        # Notações finais com Loss
        loss_note = MathTex(
            r"\mathcal{L} = \text{loss}",
            font_size=18,
            color=RED
        )
        loss_note.next_to(activation_note, RIGHT, buff=1)
        
        self.play(
            Write(weight_note),
            Write(bias_note),
            Write(activation_note),
            Write(loss_note),
            run_time=1.5
        )
        self.wait(2)
        
        # Guardar
        self.arch_group = VGroup(
            arch_title, connections, input_layer, hidden1, hidden2,
            output_layer, input_labels, output_label, layer_labels,
            bias1_nodes, bias1_labels, bias1_connections,
            bias2_nodes, bias2_labels, bias2_connections,
            bias3_node, bias3_label, bias3_connection,
            sigma1_symbols, sigma2_symbols,
            hidden1_hidden2_lines, hidden2_output_lines,
            loss_box, loss_label, loss_text, output_to_loss,
            weight_note, bias_note, activation_note, loss_note
        )

    def show_composition(self):
        """Mostra a composição EXPANDIDA neurônio por neurônio da rede 5→3→3→1"""
        
        self.play(FadeOut(self.arch_group), run_time=0.8)
        self.wait(0.5)
        
        # Título explicativo
        intro_text = Text("Expandindo cada neurônio", font_size=36, color=WHITE)
        intro_text.to_edge(UP, buff=0.3)
        self.play(Write(intro_text), run_time=1.2)
        self.wait(1)
        
        # Hidden Layer 1 - 3 neurônios (cada um recebe 5 entradas)
        h1_title = Text("Hidden Layer 1 (3 neurônios):", font_size=28, color=BLUE, weight=BOLD)
        h1_title.next_to(intro_text, DOWN, buff=0.6)
        
        h1_neurons = VGroup(
            MathTex(r"h_1 = \sigma(x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + x_5 w_{15} + b_1)", font_size=22, color=BLUE),
            MathTex(r"h_2 = \sigma(x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + x_5 w_{25} + b_2)", font_size=22, color=BLUE),
            MathTex(r"h_3 = \sigma(x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + x_5 w_{35} + b_3)", font_size=22, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        h1_neurons.next_to(h1_title, DOWN, buff=0.5)
        
        self.play(Write(h1_title), run_time=1)
        self.wait(0.5)
        for neuron in h1_neurons:
            self.play(Write(neuron), run_time=1.5)
            self.wait(0.5)
        
        self.wait(1.5)
        
        # Hidden Layer 2 - 3 neurônios (cada um recebe 3 entradas de h1)
        h2_title = Text("Hidden Layer 2 (3 neurônios):", font_size=28, color=BLUE, weight=BOLD)
        h2_title.next_to(h1_neurons, DOWN, buff=0.6)
        
        h2_neurons = VGroup(
            MathTex(r"h_4 = \sigma(h_1 w_{41} + h_2 w_{42} + h_3 w_{43} + b_4)", font_size=22, color=BLUE),
            MathTex(r"h_5 = \sigma(h_1 w_{51} + h_2 w_{52} + h_3 w_{53} + b_5)", font_size=22, color=BLUE),
            MathTex(r"h_6 = \sigma(h_1 w_{61} + h_2 w_{62} + h_3 w_{63} + b_6)", font_size=22, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        h2_neurons.next_to(h2_title, DOWN, buff=0.5)
        
        self.play(Write(h2_title), run_time=1)
        self.wait(0.5)
        for neuron in h2_neurons:
            self.play(Write(neuron), run_time=1.5)
            self.wait(0.5)
        
        self.wait(1.5)
        self.play(
            FadeOut(intro_text),
            FadeOut(h1_title),
            FadeOut(h1_neurons),
            FadeOut(h2_title),
            FadeOut(h2_neurons),
            run_time=1
        )
        self.wait(0.5)
        
        # Output e Loss - tudo em uma única expressão gigante
        output_title = Text("Saída final (1 neurônio):", font_size=28, color=ORANGE, weight=BOLD)
        output_title.to_edge(UP, buff=0.5)
        
        self.play(Write(output_title), run_time=1.2)
        self.wait(1)
        
        # Expressão da saída (sem Loss primeiro)
        output_formula = MathTex(
            r"\hat{y} = h_4 w_{71} + h_5 w_{72} + h_6 w_{73} + b_7",
            font_size=22,
            color=ORANGE
        )
        output_formula.next_to(output_title, DOWN, buff=0.6)
        
        self.play(Write(output_formula), run_time=2)
        self.wait(1.5)
        
        # Agora mostrar com Loss
        loss_title = Text("Com Loss function:", font_size=28, color=RED, weight=BOLD)
        loss_title.next_to(output_formula, DOWN, buff=0.8)
        
        loss_formula = MathTex(
            r"\mathcal{L}(\hat{y}, y) = \mathcal{L}(h_4 w_{71} + h_5 w_{72} + h_6 w_{73} + b_7, y)",
            font_size=20,
            color=RED
        )
        loss_formula.next_to(loss_title, DOWN, buff=0.5)
        
        self.play(Write(loss_title), run_time=1.2)
        self.wait(0.5)
        self.play(Write(loss_formula), run_time=2.5)
        self.wait(3)


class ActivationFunctions(Scene):
    """Mostra fórmulas de funções de ativação com explicação dos termos"""

    def construct(self):
        title = Text("Funções de Ativação", font_size=48, weight=BOLD, color=GREEN)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(1)

        # Mostrar cada função
        self.show_sigmoid()
        self.wait(1.5)
        
        self.show_relu()
        self.wait(1.5)
        
        self.show_tanh()
        self.wait(2)

    def show_sigmoid(self):
        """Mostra a função Sigmoid"""
        
        # Nome
        name = Text("Sigmoid", font_size=36, color=BLUE, weight=BOLD)
        name.shift(UP * 2)
        
        # Fórmula
        formula = MathTex(
            r"\sigma(x) = \frac{1}{1 + e^{-x}}",
            font_size=44,
            color=BLUE
        )
        formula.next_to(name, DOWN, buff=0.6)
        
        # Explicação dos termos
        terms = VGroup(
            MathTex(r"x = \text{entrada}", font_size=28),
            MathTex(r"e = \text{número de Euler } (\approx 2.718)", font_size=28),
            MathTex(r"\sigma(x) = \text{saída entre } 0 \text{ e } 1", font_size=28, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        terms.next_to(formula, DOWN, buff=0.8)
        
        # Animar
        self.play(Write(name), run_time=0.8)
        self.play(Write(formula), run_time=1.2)
        self.wait(0.5)
        self.play(Write(terms), run_time=1.5)
        self.wait(1.5)
        
        # Guardar
        self.sigmoid_group = VGroup(name, formula, terms)
        self.play(FadeOut(self.sigmoid_group), run_time=0.8)

    def show_relu(self):
        """Mostra a função ReLU"""
        
        # Nome
        name = Text("ReLU", font_size=36, color=RED, weight=BOLD)
        name.shift(UP * 2)
        
        # Fórmula
        formula = MathTex(
            r"\text{ReLU}(x) = \max(0, x)",
            font_size=44,
            color=RED
        )
        formula.next_to(name, DOWN, buff=0.6)
        
        # Equivalente
        equiv = MathTex(
            r"= \begin{cases} x & \text{se } x > 0 \\ 0 & \text{se } x \leq 0 \end{cases}",
            font_size=38,
            color=RED
        )
        equiv.next_to(formula, DOWN, buff=0.5)
        
        # Explicação dos termos
        terms = VGroup(
            MathTex(r"x = \text{entrada}", font_size=28),
            MathTex(r"\max(0, x) = \text{maior valor entre } 0 \text{ e } x", font_size=28),
            MathTex(r"\text{ReLU}(x) = \text{saída sempre } \geq 0", font_size=28, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        terms.next_to(equiv, DOWN, buff=0.8)
        
        # Animar
        self.play(Write(name), run_time=0.8)
        self.play(Write(formula), run_time=1.2)
        self.wait(0.5)
        self.play(Write(equiv), run_time=1.2)
        self.wait(0.5)
        self.play(Write(terms), run_time=1.5)
        self.wait(1.5)
        
        # Guardar
        self.relu_group = VGroup(name, formula, equiv, terms)
        self.play(FadeOut(self.relu_group), run_time=0.8)

    def show_tanh(self):
        """Mostra a função Tanh"""
        
        # Nome
        name = Text("Tanh", font_size=36, color=PURPLE, weight=BOLD)
        name.shift(UP * 2)
        
        # Fórmula
        formula = MathTex(
            r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
            font_size=40,
            color=PURPLE
        )
        formula.next_to(name, DOWN, buff=0.6)
        
        # Explicação dos termos
        terms = VGroup(
            MathTex(r"x = \text{entrada}", font_size=28),
            MathTex(r"e = \text{número de Euler } (\approx 2.718)", font_size=28),
            MathTex(r"\tanh(x) = \text{saída entre } -1 \text{ e } 1", font_size=28, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        terms.next_to(formula, DOWN, buff=0.8)
        
        # Animar
        self.play(Write(name), run_time=0.8)
        self.play(Write(formula), run_time=1.2)
        self.wait(0.5)
        self.play(Write(terms), run_time=1.5)
        self.wait(2)


class LossFunctions(Scene):
    """Mostra fórmulas de funções de perda com explicação dos termos"""

    def construct(self):
        title = Text("Funções de Perda", font_size=48, weight=BOLD, color=RED)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(1)

        # Mostrar cada função
        self.show_mse()
        self.wait(2)
        
        self.show_cross_entropy()
        self.wait(2)

    def show_mse(self):
        """Mostra Mean Squared Error"""
        
        # Nome
        name = Text("MSE - Mean Squared Error", font_size=36, color=BLUE, weight=BOLD)
        name.shift(UP * 2.2)
        
        # Fórmula
        formula = MathTex(
            r"\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            font_size=40,
            color=BLUE
        )
        formula.next_to(name, DOWN, buff=0.6)
        
        # Explicação dos termos
        terms = VGroup(
            MathTex(r"n = \text{número de amostras}", font_size=28),
            MathTex(r"y_i = \text{valor real (esperado)}", font_size=28),
            MathTex(r"\hat{y}_i = \text{valor predito pela rede}", font_size=28),
            MathTex(r"(y_i - \hat{y}_i)^2 = \text{erro quadrático}", font_size=28),
            MathTex(r"\mathcal{L}_{\text{MSE}} = \text{média dos erros}", font_size=28, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        terms.next_to(formula, DOWN, buff=0.8)
        
        # Animar
        self.play(Write(name), run_time=1)
        self.play(Write(formula), run_time=1.5)
        self.wait(0.5)
        self.play(Write(terms), run_time=2)
        self.wait(2)
        
        # Guardar
        self.mse_group = VGroup(name, formula, terms)
        self.play(FadeOut(self.mse_group), run_time=0.8)

    def show_cross_entropy(self):
        """Mostra Cross-Entropy Loss"""
        
        # Nome
        name = Text("Cross-Entropy Loss", font_size=36, color=ORANGE, weight=BOLD)
        name.shift(UP * 2.2)
        
        # Fórmula (caso binário)
        subtitle = Text("(Classificação Binária)", font_size=24, color=GRAY)
        subtitle.next_to(name, DOWN, buff=0.3)
        
        formula = MathTex(
            r"\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]",
            font_size=32,
            color=ORANGE
        )
        formula.next_to(subtitle, DOWN, buff=0.6)
        
        # Explicação dos termos
        terms = VGroup(
            MathTex(r"n = \text{número de amostras}", font_size=26),
            MathTex(r"y_i \in \{0, 1\} = \text{classe real}", font_size=26),
            MathTex(r"\hat{y}_i \in (0, 1) = \text{probabilidade predita}", font_size=26),
            MathTex(r"\log = \text{logaritmo natural (base } e)", font_size=26),
            MathTex(r"\mathcal{L}_{\text{CE}} = \text{mede divergência das probabilidades}", font_size=26, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        terms.next_to(formula, DOWN, buff=0.7)
        
        # Animar
        self.play(Write(name), run_time=1)
        self.play(Write(subtitle), run_time=0.8)
        self.play(Write(formula), run_time=1.8)
        self.wait(0.5)
        self.play(Write(terms), run_time=2)
        self.wait(2)


# ==============================================================================
# DURAÇÃO APROXIMADA:
# - NeuralNetworkAsFunction: ~70-80 segundos (função → exemplo → arquitetura → composição)
# - ActivationFunctions: ~35-40 segundos (apenas fórmulas + termos)
# - LossFunctions: ~30-35 segundos (apenas fórmulas + termos)
# TOTAL: ~2-2.5 minutos
# ==============================================================================