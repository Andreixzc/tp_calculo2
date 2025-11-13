from manim import *
import numpy as np

class ForwardPassIterations(Scene):
    def construct(self):
        # Title
        title = Text("Forward Pass - Propagação para Frente", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Network structure description
        network_desc = Text(
            "Rede Neural: 2 → 3 → 1",
            font_size=28,
            color=BLUE
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(network_desc))
        self.wait()
        
        # Create network visualization
        self.play(FadeOut(network_desc))
        
        # Input layer
        input_layer = VGroup()
        input_positions = [UP * 1, DOWN * 1]
        input_neurons = []
        for i, pos in enumerate(input_positions):
            neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.3)
            neuron.shift(LEFT * 5 + pos)
            label = MathTex(f"x_{i+1}", font_size=32).move_to(neuron)
            input_neurons.append(neuron)
            input_layer.add(VGroup(neuron, label))
        
        # Hidden layer
        hidden_layer = VGroup()
        hidden_positions = [UP * 1.5, UP * 0, DOWN * 1.5]
        hidden_neurons = []
        for i, pos in enumerate(hidden_positions):
            neuron = Circle(radius=0.3, color=GREEN, fill_opacity=0.3)
            neuron.shift(LEFT * 1 + pos)
            label = MathTex(f"h_{i+1}", font_size=32).move_to(neuron)
            hidden_neurons.append(neuron)
            hidden_layer.add(VGroup(neuron, label))
        
        # Output layer
        output_neuron = Circle(radius=0.3, color=RED, fill_opacity=0.3)
        output_neuron.shift(RIGHT * 4)
        output_label = MathTex(r"\hat{y}", font_size=32).move_to(output_neuron)
        output_layer = VGroup(output_neuron, output_label)
        
        # Create connections (edges)
        edges_input_hidden = VGroup()
        for inp in input_neurons:
            for hid in hidden_neurons:
                edge = Line(inp.get_center(), hid.get_center(), color=GRAY, stroke_width=2)
                edges_input_hidden.add(edge)
        
        edges_hidden_output = VGroup()
        for hid in hidden_neurons:
            edge = Line(hid.get_center(), output_neuron.get_center(), color=GRAY, stroke_width=2)
            edges_hidden_output.add(edge)
        
        # Draw network
        self.play(
            Create(edges_input_hidden),
            Create(edges_hidden_output),
            run_time=1
        )
        self.play(
            LaggedStart(*[Create(group) for group in input_layer], lag_ratio=0.2),
            LaggedStart(*[Create(group) for group in hidden_layer], lag_ratio=0.2),
            Create(output_layer),
            run_time=1.5
        )
        self.wait()
        
        # Show example data
        input_values = [0.5, 0.8]
        input_display = MathTex(
            f"x_1 = {input_values[0]}, \\ x_2 = {input_values[1]}",
            font_size=28,
            color=BLUE
        )
        input_display.to_edge(DOWN).shift(UP * 0.5)
        
        self.play(Write(input_display))
        self.wait(0.5)
        
        # Animate input values
        for i, val in enumerate(input_values):
            value_label = DecimalNumber(val, num_decimal_places=1, font_size=24, color=YELLOW)
            value_label.next_to(input_neurons[i], LEFT, buff=0.3)
            self.play(
                input_neurons[i].animate.set_fill(BLUE, opacity=0.8),
                FadeIn(value_label)
            )
            self.wait(0.3)
        
        # Calculate hidden layer (mock calculation)
        self.wait(0.5)
        
        # Simulate data flow to hidden layer
        hidden_values = [0.73, 0.91, 0.45]  # Mock values after activation
        for i, val in enumerate(hidden_values):
            # Highlight connections
            relevant_edges = VGroup()
            for j in range(len(input_neurons)):
                edge_idx = i + j * len(hidden_neurons)
                relevant_edges.add(edges_input_hidden[edge_idx])
            
            self.play(
                *[edge.animate.set_color(YELLOW).set_stroke(width=4) for edge in relevant_edges],
                run_time=0.3
            )
            self.play(
                hidden_neurons[i].animate.set_fill(GREEN, opacity=0.8),
                *[edge.animate.set_color(GRAY).set_stroke(width=2) for edge in relevant_edges],
                run_time=0.3
            )
            
            value_label = DecimalNumber(val, num_decimal_places=2, font_size=20, color=YELLOW)
            value_label.next_to(hidden_neurons[i], DOWN, buff=0.2)
            self.play(FadeIn(value_label))
            self.wait(0.2)
        
        # Calculate output
        self.wait(0.5)
        
        # Simulate data flow to output
        self.play(
            *[edge.animate.set_color(YELLOW).set_stroke(width=4) for edge in edges_hidden_output],
            run_time=0.4
        )
        
        output_value = 0.68  # Mock value
        self.play(
            output_neuron.animate.set_fill(RED, opacity=0.8),
            *[edge.animate.set_color(GRAY).set_stroke(width=2) for edge in edges_hidden_output],
            run_time=0.4
        )
        
        output_value_label = DecimalNumber(output_value, num_decimal_places=2, font_size=24, color=YELLOW)
        output_value_label.next_to(output_neuron, RIGHT, buff=0.3)
        self.play(FadeIn(output_value_label))
        self.wait()
        
        # Show target value and error
        result_group = VGroup()
        
        target_text = MathTex(r"y_{alvo} = 1.0", font_size=28, color=GOLD)
        target_text.to_edge(DOWN).shift(UP * 1.8)
        result_group.add(target_text)
        
        error_value = 1.0 - output_value
        error_text = MathTex(
            f"erro = {error_value:.2f}",
            font_size=28,
            color=RED
        ).next_to(target_text, DOWN, buff=0.4)
        result_group.add(error_text)
        
        self.play(Write(target_text))
        self.wait(0.3)
        self.play(Write(error_text))
        self.wait(1.5)
        
        # Clear for final question
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob != title],
            run_time=0.8
        )
        self.wait(0.5)
        
        # Big question mark appears
        big_question_mark = Text(
            "?",
            font_size=180,
            color=YELLOW,
            weight=BOLD
        )
        big_question_mark.move_to(ORIGIN)
        
        self.play(
            Write(big_question_mark),
            run_time=0.8
        )
        self.wait(0.3)
        
        # Pulse the question mark
        self.play(
            big_question_mark.animate.scale(1.2).set_color(RED),
            run_time=0.3
        )
        self.play(
            big_question_mark.animate.scale(1/1.2).set_color(YELLOW),
            run_time=0.3
        )
        self.wait(0.5)
        
        # Move question mark up and show the question
        self.play(
            big_question_mark.animate.shift(UP * 1.5).scale(0.5),
            run_time=0.6
        )
        
        # The question
        question = Text(
            "Como atualizar os pesos?",
            font_size=42,
            color=YELLOW,
            weight=BOLD
        )
        question.shift(DOWN * 0.5)
        
        self.play(
            FadeIn(question, shift=UP * 0.3),
            run_time=0.8
        )
        self.wait(1)
        
        # Hint
        hint = Text(
            "→ Backpropagation",
            font_size=36,
            color=GREEN,
            slant=ITALIC
        ).next_to(question, DOWN, buff=0.8)
        
        self.play(FadeIn(hint, shift=UP * 0.3))
        self.wait(1.5)
        
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1
        )
        self.wait()
