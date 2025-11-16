from manim import *
import numpy as np

# Configuração para aspecto widescreen 16:9
config.frame_width = 16
config.frame_height = 9
config.pixel_width = 1920
config.pixel_height = 1080

class ForwardPassIterations(Scene):
    def construct(self):
        # Title
        title = Text("Forward Pass - Propagação para Frente", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.5)
        
        # Definir funções
        subtitle = Text("Funções Utilizadas:", font_size=24, color=BLUE)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        activation_def = MathTex(
            r"\text{Ativação: } \sigma(z) = \frac{1}{1 + e^{-z}} \text{ (Sigmoid)}",
            font_size=22,
            color=GREEN
        )
        activation_def.next_to(subtitle, DOWN, buff=0.4)
        
        loss_def = MathTex(
            r"\text{Loss: } \mathcal{L} = \frac{1}{2}(\hat{y} - y)^2 \text{ (MSE)}",
            font_size=22,
            color=RED
        )
        loss_def.next_to(activation_def, DOWN, buff=0.3)
        
        self.play(Write(subtitle))
        self.wait(0.3)
        self.play(Write(activation_def))
        self.wait(0.5)
        self.play(Write(loss_def))
        self.wait(1.2)
        
        # Fade out definitions
        self.play(
            FadeOut(subtitle),
            FadeOut(activation_def),
            FadeOut(loss_def),
            run_time=0.6
        )
        
        # Exemplo da casa
        house_title = Text("Exemplo: Previsão de Preço de Casa", font_size=26, color=BLUE)
        house_title.next_to(title, DOWN, buff=0.5)
        
        house_inputs = VGroup(
            Text("Área: 120m²", font_size=22, color=GREEN),
            Text("Quartos: 3", font_size=22, color=GREEN)
        ).arrange(RIGHT, buff=1.5)
        house_inputs.next_to(house_title, DOWN, buff=0.5)
        
        house_target = Text("Preço Real: R$ 200k", font_size=22, color=GOLD)
        house_target.next_to(house_inputs, DOWN, buff=0.5)

        # Normalização explícita das entradas
        norm_title = Text("Normalização das Entradas", font_size=24, color=BLUE)
        norm_title.next_to(house_target, DOWN, buff=0.6)

        norm_eq1 = MathTex(
            r"x_1 = \frac{\text{Área}}{200} = \frac{120}{200} = 0{,}60",
            font_size=22,
            color=WHITE
        )
        norm_eq1.next_to(norm_title, DOWN, buff=0.3)

        norm_eq2 = MathTex(
            r"x_2 = \frac{\text{Quartos}}{4} = \frac{3}{4} = 0{,}75",
            font_size=22,
            color=WHITE
        )
        norm_eq2.next_to(norm_eq1, DOWN, buff=0.2)
        
        self.play(Write(house_title))
        self.wait(0.3)
        self.play(Write(house_inputs))
        self.wait(0.5)
        self.play(Write(house_target))
        self.wait(0.5)
        self.play(Write(norm_title))
        self.play(Write(norm_eq1), Write(norm_eq2))
        self.wait(1.2)
        
        # Fade out example
        self.play(
            FadeOut(house_title),
            FadeOut(house_inputs),
            FadeOut(house_target),
            FadeOut(norm_title),
            FadeOut(norm_eq1),
            FadeOut(norm_eq2),
            run_time=0.6
        )
        
        # Exemplo simples: 2 inputs -> 2 hidden -> 1 output
        # Valores normalizados da casa
        # 120m² / 200 = 0.60   e   3 quartos / 4 = 0.75
        
        # Posições
        input_y_positions = [UP * 1.2, DOWN * 1.2]
        hidden_y_positions = [UP * 1.2, DOWN * 1.2]
        
        # ========== INPUTS ==========
        input_neurons = VGroup()
        input_values = [0.6, 0.75]  # Valores normalizados (x1 = 0.60, x2 = 0.75)
        input_labels_text = ["Área", "Quartos"]
        
        for y_pos, val, label_text in zip(input_y_positions, input_values, input_labels_text):
            neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.3)
            neuron.shift(LEFT * 5.5 + y_pos)
            label = Text(label_text, font_size=18).move_to(neuron)
            
            # Valor ao lado do neurônio
            value_text = Text(f"{val:.2f}", font_size=20, color=YELLOW)
            value_text.next_to(neuron, LEFT, buff=0.4)
            
            input_neurons.add(VGroup(neuron, label, value_text))
        
        # ========== HIDDEN LAYER ==========
        hidden_neurons = VGroup()
        hidden_labels_text = ["h₁", "h₂"]
        
        for y_pos, label_text in zip(hidden_y_positions, hidden_labels_text):
            neuron = Circle(radius=0.3, color=GREEN, fill_opacity=0.3)
            neuron.shift(LEFT * 1.5 + y_pos)
            label = Text(label_text, font_size=24).move_to(neuron)
            hidden_neurons.add(VGroup(neuron, label))
        
        # Bias para hidden layer
        bias_hidden = VGroup()
        # Mantemos os mesmos biases do exemplo original
        bias_hidden_values = [0.1, -0.2]
        for h_group, b_val in zip(hidden_neurons, bias_hidden_values):
            h_neuron = h_group[0]
            bias_pos = h_neuron.get_center() + UP * 0.8
            bias_circle = Circle(radius=0.15, color=PURPLE, fill_opacity=0.5)
            bias_circle.move_to(bias_pos)
            bias_label = Text(f"b={b_val:.1f}", font_size=14, color=PURPLE)
            bias_label.next_to(bias_circle, UP, buff=0.15)
            bias_hidden.add(VGroup(bias_circle, bias_label))
        
        # ========== OUTPUT ==========
        output_neuron = Circle(radius=0.3, color=ORANGE, fill_opacity=0.3)
        output_neuron.shift(RIGHT * 3.5)
        output_label = Text("ŷ", font_size=24).move_to(output_neuron)
        output_group = VGroup(output_neuron, output_label)
        
        # Bias para output (ajustado para melhorar a predição)
        bias_output_val = -0.7
        bias_output_pos = output_neuron.get_center() + UP * 0.8
        bias_output_circle = Circle(radius=0.15, color=PURPLE, fill_opacity=0.5)
        bias_output_circle.move_to(bias_output_pos)
        bias_output_label = Text(f"b={bias_output_val:.1f}", font_size=14, color=PURPLE)
        bias_output_label.next_to(bias_output_circle, UP, buff=0.15)
        bias_output_group = VGroup(bias_output_circle, bias_output_label)
        
        # ========== CONEXÕES INPUT -> HIDDEN ==========
        edges_input_hidden = VGroup()
        # Mesmos pesos da camada escondida do exemplo original
        weights_input_hidden = [
            [0.5, 0.3],   # Pesos de Área para h1, h2
            [0.4, -0.2]   # Pesos de Quartos para h1, h2
        ]
        
        weight_labels_ih = VGroup()
        for i, input_group in enumerate(input_neurons):
            for j, hidden_group in enumerate(hidden_neurons):
                start = input_group[0].get_right()
                end = hidden_group[0].get_left()
                edge = Line(start, end, color=GRAY, stroke_width=1.5, stroke_opacity=0.5)
                edges_input_hidden.add(edge)
                
                # Label do peso
                weight = weights_input_hidden[i][j]
                mid_point = (start + end) / 2
                w_label = Text(f"w={weight:.1f}", font_size=12, color=RED)
                w_label.move_to(mid_point)
                w_label.shift(UP * 0.25 if j == 0 else DOWN * 0.25)
                weight_labels_ih.add(w_label)
        
        # ========== CONEXÕES HIDDEN -> OUTPUT ==========
        edges_hidden_output = VGroup()
        # Pesos ajustados para gerar uma predição próxima de R$ 200k
        # h1 ≈ 0.67, h2 ≈ 0.46  ->  ŷ_norm ≈ 0.38  ->  preço ≈ 0.38 * 500k ≈ 190k
        weights_hidden_output = [1.0, 0.9]
        
        weight_labels_ho = VGroup()
        for i, (hidden_group, weight) in enumerate(zip(hidden_neurons, weights_hidden_output)):
            start = hidden_group[0].get_right()
            end = output_neuron.get_left()
            edge = Line(start, end, color=GRAY, stroke_width=1.5, stroke_opacity=0.5)
            edges_hidden_output.add(edge)
            
            # Label do peso
            mid_point = (start + end) / 2
            w_label = Text(f"w={weight:.1f}", font_size=12, color=RED)
            w_label.move_to(mid_point)
            w_label.shift(UP * 0.25 if i == 0 else DOWN * 0.25)
            weight_labels_ho.add(w_label)
        
        # ========== BIAS ARROWS ==========
        bias_arrows_hidden = VGroup()
        for bias_group, hidden_group in zip(bias_hidden, hidden_neurons):
            arrow = Arrow(
                bias_group[0].get_bottom(),
                hidden_group[0].get_top(),
                color=PURPLE,
                stroke_width=1.5,
                buff=0.1,
                max_tip_length_to_length_ratio=0.2
            )
            bias_arrows_hidden.add(arrow)
        
        bias_arrow_output = Arrow(
            bias_output_circle.get_bottom(),
            output_neuron.get_top(),
            color=PURPLE,
            stroke_width=1.5,
            buff=0.1,
            max_tip_length_to_length_ratio=0.2
        )
        
        # Símbolos de ativação (sigma)
        sigma_hidden = VGroup()
        for hidden_group in hidden_neurons:
            sigma = MathTex(r"\sigma", font_size=18, color=GOLD)
            sigma.move_to(hidden_group[0].get_right() + RIGHT * 0.4)
            sigma_hidden.add(sigma)
        
        # ========== ANIMAÇÃO ==========
        
        # Desenhar rede
        self.play(
            LaggedStart(*[Create(group) for group in input_neurons], lag_ratio=0.3),
            run_time=1.2
        )
        self.wait(0.5)
        
        self.play(
            Create(edges_input_hidden),
            run_time=0.8
        )
        self.play(Write(weight_labels_ih), run_time=0.8)
        self.wait(0.5)
        
        self.play(
            Create(bias_hidden),
            run_time=0.8
        )
        self.play(Create(bias_arrows_hidden), run_time=0.6)
        self.wait(0.3)
        
        self.play(
            LaggedStart(*[Create(group[:-1]) for group in hidden_neurons], lag_ratio=0.3),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(Write(sigma_hidden), run_time=0.6)
        self.wait(0.5)
        
        self.play(Create(edges_hidden_output), run_time=0.8)
        self.play(Write(weight_labels_ho), run_time=0.8)
        self.wait(0.5)
        
        self.play(
            Create(bias_output_group),
            run_time=0.8
        )
        self.play(Create(bias_arrow_output), run_time=0.6)
        self.wait(0.3)
        
        self.play(Create(output_group), run_time=1)
        self.wait(0.8)
        
        # ========== FORWARD PASS DINÂMICO ==========
        
        # Calcular valores (usando sigmoid)
        # h1 = σ(0.6*0.5 + 0.75*0.4 + 0.1) = σ(0.3 + 0.3 + 0.1) = σ(0.7) ≈ 0.67
        # h2 = σ(0.6*0.3 + 0.75*(-0.2) - 0.2) = σ(0.18 - 0.15 - 0.2) = σ(-0.17) ≈ 0.46
        # ŷ_norm ≈ 1.0*h1 + 0.9*h2 - 0.7 ≈ 0.38
        # Preço previsto ≈ 0.38 * 500k ≈ R$ 190k
        
        hidden_values = [0.67, 0.46]
        output_value_norm = 0.38  # Normalizado (0.38 * 500k ≈ 190k)
        
        # Propagar h1
        self.play(
            hidden_neurons[0][0].animate.set_fill(GREEN, opacity=0.7),
            edges_input_hidden[0].animate.set_color(YELLOW).set_stroke(width=3),
            edges_input_hidden[2].animate.set_color(YELLOW).set_stroke(width=3),
            bias_arrows_hidden[0].animate.set_color(YELLOW).set_stroke(width=2.5),
            run_time=0.5
        )
        
        h1_value = Text(f"{hidden_values[0]:.2f}", font_size=18, color=YELLOW)
        h1_value.next_to(hidden_neurons[0][0], RIGHT, buff=0.3)
        self.play(FadeIn(h1_value), run_time=0.4)
        
        self.play(
            edges_input_hidden[0].animate.set_color(GRAY).set_stroke(width=1.5),
            edges_input_hidden[2].animate.set_color(GRAY).set_stroke(width=1.5),
            bias_arrows_hidden[0].animate.set_color(PURPLE).set_stroke(width=1.5),
            run_time=0.3
        )
        
        # Propagar h2
        self.play(
            hidden_neurons[1][0].animate.set_fill(GREEN, opacity=0.7),
            edges_input_hidden[1].animate.set_color(YELLOW).set_stroke(width=3),
            edges_input_hidden[3].animate.set_color(YELLOW).set_stroke(width=3),
            bias_arrows_hidden[1].animate.set_color(YELLOW).set_stroke(width=2.5),
            run_time=0.5
        )
        
        h2_value = Text(f"{hidden_values[1]:.2f}", font_size=18, color=YELLOW)
        h2_value.next_to(hidden_neurons[1][0], RIGHT, buff=0.3)
        self.play(FadeIn(h2_value), run_time=0.4)
        
        self.play(
            edges_input_hidden[1].animate.set_color(GRAY).set_stroke(width=1.5),
            edges_input_hidden[3].animate.set_color(GRAY).set_stroke(width=1.5),
            bias_arrows_hidden[1].animate.set_color(PURPLE).set_stroke(width=1.5),
            run_time=0.3
        )
        
        self.wait(0.5)
        
        # Propagar para output
        self.play(
            output_neuron.animate.set_fill(ORANGE, opacity=0.7),
            edges_hidden_output[0].animate.set_color(YELLOW).set_stroke(width=3),
            edges_hidden_output[1].animate.set_color(YELLOW).set_stroke(width=3),
            bias_arrow_output.animate.set_color(YELLOW).set_stroke(width=2.5),
            run_time=0.5
        )
        
        y_value_norm = Text(f"{output_value_norm:.2f}", font_size=18, color=YELLOW)
        y_value_norm.next_to(output_neuron, RIGHT, buff=0.3)
        self.play(FadeIn(y_value_norm), run_time=0.4)
        
        self.play(
            edges_hidden_output[0].animate.set_color(GRAY).set_stroke(width=1.5),
            edges_hidden_output[1].animate.set_color(GRAY).set_stroke(width=1.5),
            bias_arrow_output.animate.set_color(PURPLE).set_stroke(width=1.5),
            run_time=0.3
        )
        
        self.wait(0.8)
        
        # ========== RESULTADO E LOSS ==========
        
        # Mostrar predição vs real
        prediction_price = 190  # R$ 190k (próximo de 200k)
        real_price = 200        # R$ 200k
        
        result_box = VGroup(
            Text(f"Predição: R$ {prediction_price}k", font_size=24, color=ORANGE),
            Text(f"Real: R$ {real_price}k", font_size=24, color=GOLD),
        ).arrange(DOWN, buff=0.4)
        result_box.to_edge(DOWN, buff=1.5)
        
        self.play(Write(result_box[0]))
        self.wait(0.5)
        self.play(Write(result_box[1]))
        self.wait(0.8)
        
        # Calcular e mostrar loss (usando valores normalizados de preço)
        # y_hat_norm = 190 / 500 = 0.38
        # y_norm = 200 / 500 = 0.40
        # L = 1/2 (0.38 - 0.40)^2 ≈ 0.0002
        
        loss_value = 0.5 * ((prediction_price - real_price) / 500) ** 2  # Normalizado
        
        loss_text = MathTex(
            r"\mathcal{L} = \frac{1}{2}(0{,}38 - 0{,}40)^2 \approx 0{,}0002",
            font_size=22,
            color=RED
        )
        loss_text.next_to(result_box, DOWN, buff=0.5)
        
        self.play(Write(loss_text))
        self.wait(1.5)
        
        # Conclusão rápida dentro da cena atual
        conclusion = Text(
            "Erro pequeno mas ainda presente",
            font_size=22,
            color=RED
        ).to_edge(DOWN, buff=0.3)
        
        self.play(
            result_box.animate.shift(UP * 0.5),
            loss_text.animate.shift(UP * 0.5),
            run_time=0.5
        )
        self.play(Write(conclusion))
        self.wait(1.5)
        
        # ========== NOVA SEÇÃO  TELA DE PERGUNTA ==========
        
        # Some com tudo para abrir uma nova "tela"
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
        self.wait(0.3)
        
        # Fundo tipo tela
        question_bg = Rectangle(
            width=14,
            height=7,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_color=BLUE,
            stroke_width=2
        )
        question_bg.move_to(ORIGIN)
        
        question_title = Text(
            "Predição de preço vs valor real",
            font_size=30,
            color=WHITE
        ).move_to(UP * 3)
        
        # Mostra de novo os valores, agora em destaque
        pred_text_big = Text(
            f"Predição da rede  R$ {prediction_price}k",
            font_size=28,
            color=ORANGE
        )
        real_text_big = Text(
            f"Valor real  R$ {real_price}k",
            font_size=28,
            color=GOLD
        )
        values_group = VGroup(pred_text_big, real_text_big).arrange(DOWN, buff=0.3)
        values_group.move_to(DOWN * 0.5 + LEFT * 3)
        
        # Pergunta
        question_text = Text(
            "Como ajustar os pesos para diminuir o erro?",
            font_size=32,
            color=YELLOW
        ).move_to(UP * 0.5)
        
        # Resposta
        answer_label = Text(
            "Resposta:",
            font_size=28,
            color=WHITE
        ).move_to(DOWN * 1)
        
        answer_text = Text(
            "Backpropagation",
            font_size=36,
            color=GREEN,
            weight=BOLD
        ).next_to(answer_label, DOWN, buff=0.5)
        
        pergunta_group = VGroup(
            question_bg,
            question_title,
            values_group,
            question_text,
            answer_label,
            answer_text
        )
        
        # Animação da tela de pergunta
        self.play(FadeIn(question_bg), run_time=0.6)
        self.play(Write(question_title), run_time=0.6)
        self.play(
            LaggedStart(
                FadeIn(pred_text_big),
                FadeIn(real_text_big),
                lag_ratio=0.3
            ),
            run_time=0.8
        )
        self.play(Write(question_text), run_time=0.8)
        self.wait(1)
        self.play(Write(answer_label), run_time=0.5)
        self.play(Write(answer_text), run_time=0.8)
        
        self.wait(3)
        
        # Final da cena
        self.play(FadeOut(pergunta_group), run_time=0.8)
        self.wait(0.5)
