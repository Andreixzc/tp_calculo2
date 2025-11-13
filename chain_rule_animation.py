from manim import *


class ChainRule(Scene):
    """Regra da cadeia - intuição com funções compostas (simplificado)"""

    def construct(self):
        title = Text("Regra da Cadeia", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show function flow
        self.show_function_flow()
        self.wait(2)

        # Show chain rule unfolded
        self.show_chain_rule_unfolded()
        self.wait(3)

    def show_function_flow(self):
        """Show x → f(x) → g(x) → h(x) = z"""

        flow_title = Text(
            "Composição de Funções:", font_size=30, weight=BOLD, color=YELLOW
        )
        flow_title.shift(UP * 2.5)

        # Function boxes (simplified - just x everywhere)
        x_box = Rectangle(height=0.8, width=1, color=BLUE, stroke_width=2).shift(
            LEFT * 5.5
        )
        x_label = MathTex("x", font_size=32, color=BLUE).move_to(x_box)

        f_box = Rectangle(height=0.8, width=1.5, color=GREEN, stroke_width=2).shift(
            LEFT * 3
        )
        f_label = MathTex("f(x)", font_size=28, color=GREEN).move_to(f_box)

        g_box = Rectangle(height=0.8, width=1.5, color=ORANGE, stroke_width=2).shift(
            RIGHT * 0
        )
        g_label = MathTex("g(x)", font_size=28, color=ORANGE).move_to(g_box)

        h_box = Rectangle(height=0.8, width=1.5, color=PURPLE, stroke_width=2).shift(
            RIGHT * 3
        )
        h_label = MathTex("h(x)", font_size=28, color=PURPLE).move_to(h_box)

        z_box = Rectangle(height=0.8, width=1, color=RED, stroke_width=2).shift(
            RIGHT * 5.5
        )
        z_label = MathTex("z", font_size=32, color=RED).move_to(z_box)

        # Arrows
        arrow1 = Arrow(
            x_box.get_right(),
            f_box.get_left(),
            color=WHITE,
            stroke_width=2,
            buff=0.1,
        )
        arrow2 = Arrow(
            f_box.get_right(),
            g_box.get_left(),
            color=WHITE,
            stroke_width=2,
            buff=0.1,
        )
        arrow3 = Arrow(
            g_box.get_right(),
            h_box.get_left(),
            color=WHITE,
            stroke_width=2,
            buff=0.1,
        )
        arrow4 = Arrow(
            h_box.get_right(),
            z_box.get_left(),
            color=WHITE,
            stroke_width=2,
            buff=0.1,
        )

        self.play(Write(flow_title), run_time=1)
        self.wait(1)

        # Build flow
        self.play(Create(x_box), Write(x_label), run_time=1)
        self.wait(0.5)
        self.play(Create(arrow1), run_time=0.8)
        self.wait(0.3)
        self.play(Create(f_box), Write(f_label), run_time=1.2)
        self.wait(0.5)
        self.play(Create(arrow2), run_time=0.8)
        self.wait(0.3)
        self.play(Create(g_box), Write(g_label), run_time=1.2)
        self.wait(0.5)
        self.play(Create(arrow3), run_time=0.8)
        self.wait(0.3)
        self.play(Create(h_box), Write(h_label), run_time=1.2)
        self.wait(0.5)
        self.play(Create(arrow4), run_time=0.8)
        self.wait(0.3)
        self.play(Create(z_box), Write(z_label), run_time=1)
        self.wait(2)

        self.flow_group = VGroup(
            x_box,
            x_label,
            arrow1,
            f_box,
            f_label,
            arrow2,
            g_box,
            g_label,
            arrow3,
            h_box,
            h_label,
            arrow4,
            z_box,
            z_label,
        )
        self.flow_title = flow_title

    def show_chain_rule_unfolded(self):
        """Show unfolded chain rule for each function"""

        # Move flow up
        self.play(
            self.flow_group.animate.scale(0.7).shift(UP * 0.8),
            self.flow_title.animate.shift(UP * 0.5),
            run_time=2,
        )
        self.wait(1)

        # Question
        question = Text(
            "Como calcular a influência de cada função em z?",
            font_size=26,
            color=YELLOW,
        ).shift(DOWN * 0.5)

        self.play(Write(question), run_time=2)
        self.wait(2)

        # Chain rule title
        rule_title = Text("Regra da Cadeia:", font_size=28, weight=BOLD, color=GREEN)
        rule_title.shift(DOWN * 1.8)

        self.play(Write(rule_title), run_time=1)
        self.wait(1)

        # Unfolded chain rule
        chain_rule = MathTex(
            r"\frac{dz}{dx} = \frac{dz}{dh} \cdot \frac{dh}{dg} \cdot \frac{dg}{df} \cdot \frac{df}{dx}",
            font_size=38,
            color=GREEN,
        ).next_to(rule_title, DOWN, buff=0.5)

        self.play(Write(chain_rule), run_time=2.5)
        self.wait(2)

        # Individual derivatives
        individual = (
            VGroup(
                MathTex(r"\frac{dh}{dx} = h'(x)", font_size=28, color=PURPLE),
                MathTex(r"\frac{dg}{dx} = g'(x)", font_size=28, color=ORANGE),
                MathTex(r"\frac{df}{dx} = f'(x)", font_size=28, color=GREEN),
            )
            .arrange(RIGHT, buff=0.8)
            .next_to(chain_rule, DOWN, buff=0.7)
        )

        self.play(
            LaggedStart(*[Write(eq) for eq in individual], lag_ratio=0.5),
            run_time=3,
        )
        self.wait(2.5)
