from manim import *


class LossFunction(Scene):
    """Mostra valor da rede sendo plugado na loss function"""

    def construct(self):
        title = Text("Função de Perda (Loss)", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show network outputting value
        self.show_network_output()
        self.wait(2)

        # Plug into loss
        self.plug_into_loss()
        self.wait(3)

    def show_network_output(self):
        """Show simplified network with output"""

        # Simple vertical network (left side)
        network_x = -3.5

        inputs = (
            VGroup(
                Circle(radius=0.2, color=BLUE, fill_opacity=0.3),
                Circle(radius=0.2, color=BLUE, fill_opacity=0.3),
            )
            .arrange(RIGHT, buff=0.5)
            .shift(LEFT * network_x + UP * 2)
        )

        hidden = (
            VGroup(
                Circle(radius=0.2, color=YELLOW, fill_opacity=0.2),
                Circle(radius=0.2, color=YELLOW, fill_opacity=0.2),
                Circle(radius=0.2, color=YELLOW, fill_opacity=0.2),
            )
            .arrange(RIGHT, buff=0.4)
            .shift(LEFT * network_x + UP * 0.5)
        )

        output = Circle(radius=0.25, color=GREEN, fill_opacity=0.3).shift(
            LEFT * network_x + DOWN * 1
        )
        output_value = MathTex(r"\hat{y} = 0.23", font_size=28, color=GREEN).next_to(
            output, DOWN, buff=0.3
        )

        # Connections (simplified)
        input_hidden_lines = VGroup()
        for inp in inputs:
            for hid in hidden:
                line = Line(
                    inp.get_bottom(),
                    hid.get_top(),
                    color=GRAY,
                    stroke_width=1,
                    stroke_opacity=0.4,
                )
                input_hidden_lines.add(line)

        hidden_output_lines = VGroup()
        for hid in hidden:
            line = Line(
                hid.get_bottom(),
                output.get_top(),
                color=GRAY,
                stroke_width=1,
                stroke_opacity=0.4,
            )
            hidden_output_lines.add(line)

        network_label = Text("Rede Neural", font_size=24).next_to(inputs, UP, buff=0.5)

        # Animate
        self.play(Write(network_label), run_time=1)
        self.wait(0.5)
        self.play(
            LaggedStart(*[Create(inp) for inp in inputs], lag_ratio=0.2),
            run_time=1.5,
        )
        self.wait(0.5)
        self.play(Create(input_hidden_lines), run_time=1.5)
        self.wait(0.3)
        self.play(
            LaggedStart(*[Create(h) for h in hidden], lag_ratio=0.2),
            run_time=1.5,
        )
        self.wait(0.5)
        self.play(Create(hidden_output_lines), run_time=1.5)
        self.wait(0.3)
        self.play(Create(output), run_time=1)
        self.wait(0.5)
        self.play(Write(output_value), run_time=1.5)
        self.wait(1.5)

        self.network_group = VGroup(
            inputs,
            hidden,
            output,
            output_value,
            input_hidden_lines,
            hidden_output_lines,
            network_label,
        )
        self.output = output
        self.output_value_text = output_value

    def plug_into_loss(self):
        """Show output being plugged into loss function"""

        # Real value
        real_value = MathTex("y = 1", font_size=32, color=BLUE).shift(
            RIGHT * 3.5 + UP * 2
        )
        real_label = Text("(Valor Real)", font_size=20, slant=ITALIC).next_to(
            real_value, DOWN, buff=0.2
        )

        self.play(Write(real_value), Write(real_label), run_time=1.5)
        self.wait(1.5)

        # Arrow from network to loss formula
        arrow = Arrow(
            self.output.get_right(),
            RIGHT * 0.5 + DOWN * 1,
            color=ORANGE,
            stroke_width=3,
        )

        # Loss function formula
        loss_formula = MathTex(
            r"L = \frac{1}{2}(y - \hat{y})^2", font_size=40, color=RED
        ).shift(RIGHT * 2.5 + DOWN * 1)

        self.play(Create(arrow), run_time=1.5)
        self.wait(0.5)
        self.play(Write(loss_formula), run_time=2)
        self.wait(2)

        # Substitute values
        substitution = MathTex(
            r"L = \frac{1}{2}(1 - 0.23)^2", font_size=36, color=ORANGE
        ).next_to(loss_formula, DOWN, buff=0.7)

        result = MathTex(r"L = 0.296", font_size=40, color=RED).next_to(
            substitution, DOWN, buff=0.6
        )

        error_text = Text("Erro grande!", font_size=32, color=RED, weight=BOLD).next_to(
            result, DOWN, buff=0.6
        )

        self.play(Write(substitution), run_time=2)
        self.wait(2)
        self.play(Write(result), run_time=1.5)
        self.wait(1.5)
        self.play(Write(error_text), run_time=1)
        self.wait(2.5)
