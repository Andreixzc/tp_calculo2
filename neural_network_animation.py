from manim import *
import numpy as np

# ==============================================================================
# 7 ANIMAÇÕES - REDES NEURAIS E GRADIENT DESCENT
#
# 1. NeuralNetworkArchitecture - Arquitetura da rede multicamada
# 2. LossFunction - Função de perda e o problema de otimização
# 3. GradientDescent2D - Intuição geométrica em 2D
# 4. GradientDescent3D - Visualização em superfície 3D
# 5. ChainRule - Regra da cadeia do cálculo
# 6. Backpropagation - Propagação do gradiente para trás
# 7. XORTraining - Treinamento completo resolvendo XOR
#
# Renderizar (teste):
#   manim -pql neural_network_animation.py NeuralNetworkArchitecture
#   manim -pql neural_network_animation.py LossFunction
#   ... etc
#
# Renderizar (alta qualidade):
#   manim -pqh neural_network_animation.py NeuralNetworkArchitecture
# ==============================================================================


class NeuralNetworkArchitecture(Scene):
    """Arquitetura da rede neural - layout horizontal (left to right)"""

    def construct(self):
        title = Text("Arquitetura da Rede Neural", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Build horizontal network
        self.build_horizontal_network()
        self.wait(2)

        # Show formulas
        self.show_formulas()
        self.wait(3)

    def build_horizontal_network(self):
        """Build network with horizontal layout - inputs on LEFT, loss on RIGHT"""

        # Input layer (left side)
        inputs = VGroup(
            Circle(radius=0.25, color=BLUE, fill_opacity=0.3).shift(
                LEFT * 5 + UP * 0.8
            ),
            Circle(radius=0.25, color=BLUE, fill_opacity=0.3).shift(
                LEFT * 5 + DOWN * 0.8
            ),
        )
        input_labels = VGroup(
            MathTex("x_1", font_size=24).next_to(inputs[0], LEFT, buff=0.15),
            MathTex("x_2", font_size=24).next_to(inputs[1], LEFT, buff=0.15),
        )
        input_text = Text("Entrada", font_size=22).next_to(inputs, DOWN, buff=0.6)

        # Hidden layer (center-left)
        hidden = VGroup(
            Circle(radius=0.25, color=YELLOW, fill_opacity=0.2).shift(
                LEFT * 1.5 + UP * 1.2
            ),
            Circle(radius=0.25, color=YELLOW, fill_opacity=0.2).shift(LEFT * 1.5),
            Circle(radius=0.25, color=YELLOW, fill_opacity=0.2).shift(
                LEFT * 1.5 + DOWN * 1.2
            ),
        )
        hidden_text = Text("Camada Oculta", font_size=22).next_to(
            hidden, DOWN, buff=0.6
        )

        # Output layer (center-right)
        output = Circle(radius=0.25, color=GREEN, fill_opacity=0.3).shift(RIGHT * 2.5)
        output_label = MathTex(r"\hat{y}", font_size=24).next_to(
            output, RIGHT, buff=0.15
        )
        output_text = Text("Saída", font_size=22).next_to(output, DOWN, buff=0.6)

        # Loss (right side)
        loss_box = Rectangle(height=0.6, width=1, color=RED, stroke_width=2).shift(
            RIGHT * 5.5
        )
        loss_label = MathTex("L", font_size=28, color=RED).move_to(loss_box)
        loss_text = Text("Loss", font_size=22).next_to(loss_box, DOWN, buff=0.4)

        # Connections input -> hidden (with weight labels)
        input_hidden_lines = VGroup()
        weight_labels_1 = VGroup()
        for i, inp in enumerate(inputs):
            for j, hid in enumerate(hidden):
                line = Line(
                    inp.get_right(),
                    hid.get_left(),
                    color=GRAY,
                    stroke_width=1.5,
                    stroke_opacity=0.6,
                )
                input_hidden_lines.add(line)

                # Just show one weight label for clarity
                if i == 0 and j == 1:
                    w_label = MathTex("w_{ij}", font_size=20, color=RED).move_to(
                        line.get_center() + UP * 0.3
                    )
                    weight_labels_1.add(w_label)

        # Connections hidden -> output
        hidden_output_lines = VGroup()
        weight_labels_2 = VGroup()
        for i, hid in enumerate(hidden):
            line = Line(
                hid.get_right(),
                output.get_left(),
                color=GRAY,
                stroke_width=1.5,
                stroke_opacity=0.6,
            )
            hidden_output_lines.add(line)

            if i == 1:
                w_label = MathTex("w_j", font_size=20, color=RED).move_to(
                    line.get_center() + UP * 0.3
                )
                weight_labels_2.add(w_label)

        # Output -> Loss
        output_loss_line = Line(
            output.get_right(), loss_box.get_left(), color=RED, stroke_width=2
        )

        # Animate
        self.play(
            LaggedStart(
                *[Create(inp) for inp in inputs],
                *[Write(label) for label in input_labels],
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.play(Write(input_text), run_time=1)
        self.wait(1)

        self.play(
            Create(input_hidden_lines),
            (Write(weight_labels_1[0]) if len(weight_labels_1) > 0 else Wait(0.1)),
            run_time=2,
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*[Create(h) for h in hidden], lag_ratio=0.2),
            run_time=2,
        )
        self.play(Write(hidden_text), run_time=1)
        self.wait(1)

        self.play(
            Create(hidden_output_lines),
            (Write(weight_labels_2[0]) if len(weight_labels_2) > 0 else Wait(0.1)),
            run_time=2,
        )
        self.wait(0.5)

        self.play(Create(output), Write(output_label), run_time=1.5)
        self.play(Write(output_text), run_time=1)
        self.wait(1)

        self.play(Create(output_loss_line), run_time=1.5)
        self.wait(0.5)

        self.play(Create(loss_box), Write(loss_label), run_time=1.5)
        self.play(Write(loss_text), run_time=1)
        self.wait(1.5)

        self.network = VGroup(
            inputs,
            input_labels,
            input_text,
            input_hidden_lines,
            weight_labels_1,
            hidden,
            hidden_text,
            hidden_output_lines,
            weight_labels_2,
            output,
            output_label,
            output_text,
            output_loss_line,
            loss_box,
            loss_label,
            loss_text,
        )

    def show_formulas(self):
        """Show node computation formulas and learning rate"""

        # Move network up
        self.play(self.network.animate.scale(0.75).shift(UP * 1.2), run_time=2)
        self.wait(1)

        # Formulas
        formula_title = Text(
            "Computação dos Nodes:", font_size=28, weight=BOLD, color=YELLOW
        )
        formula_title.shift(DOWN * 1.2)

        formulas = (
            VGroup(
                MathTex(
                    r"h_j = \sigma\left(\sum_i w_{ij} x_i + b_j\right)",
                    font_size=28,
                ),
                MathTex(
                    r"\hat{y} = \sigma\left(\sum_j w_j h_j + b\right)",
                    font_size=28,
                ),
                MathTex(r"L = \frac{1}{2}(y - \hat{y})^2", font_size=28, color=RED),
            )
            .arrange(DOWN, buff=0.5)
            .next_to(formula_title, DOWN, buff=0.5)
        )

        # Learning rate
        lr_text = (
            VGroup(
                Text("Learning Rate:", font_size=24, weight=BOLD, color=GREEN),
                MathTex(r"\eta \text{ ou } \alpha", font_size=28, color=GREEN),
            )
            .arrange(RIGHT, buff=0.3)
            .next_to(formulas, DOWN, buff=0.6)
        )

        self.play(Write(formula_title), run_time=1)
        self.wait(1)
        self.play(Write(formulas[0]), run_time=2)
        self.wait(2)
        self.play(Write(formulas[1]), run_time=2)
        self.wait(2)
        self.play(Write(formulas[2]), run_time=2)
        self.wait(2)
        self.play(Write(lr_text), run_time=1.5)
        self.wait(2.5)


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

        result = MathTex(r"L = 0.296", font_size=40, color=RED, weight=BOLD).next_to(
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


class GradientDescent2D(Scene):
    """Gradient descent em 2D - curva simples"""

    def construct(self):
        title = Text("Gradient Descent - 2D", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show curve
        self.show_curve()
        self.wait(2)

        # Show descent
        self.show_descent()
        self.wait(3)

    def show_curve(self):
        """Show loss curve"""

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 5, 1],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)

        labels = VGroup(
            MathTex("w", font_size=36).next_to(axes.x_axis.get_end(), RIGHT, buff=0.2),
            MathTex("L(w)", font_size=36).next_to(axes.y_axis.get_end(), UP, buff=0.2),
        )

        # Parabola
        curve = axes.plot(lambda x: 0.5 * x**2 + 0.5, color=BLUE, stroke_width=4)

        # Minimum point
        min_dot = Dot(axes.c2p(0, 0.5), color=GREEN, radius=0.1)
        min_label = Text("Mínimo", font_size=24, color=GREEN).next_to(
            min_dot, DOWN, buff=0.3
        )

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)
        self.play(Create(curve), run_time=2)
        self.wait(1)
        self.play(Create(min_dot), Write(min_label), run_time=1.5)
        self.wait(1.5)

        self.axes = axes
        self.curve = curve

    def show_descent(self):
        """Animate gradient descent steps"""

        # Starting point
        x_start = 2.5
        point = Dot(
            self.axes.c2p(x_start, 0.5 * x_start**2 + 0.5),
            color=RED,
            radius=0.12,
        )
        point_label = MathTex("w_0", font_size=28, color=RED).next_to(
            point, UP, buff=0.2
        )

        self.play(Create(point), Write(point_label), run_time=1.5)
        self.wait(1)

        # Show derivative - MOVED TO TOP to avoid overlap
        derivative_text = MathTex(
            r"\frac{dL}{dw} < 0 \Rightarrow \text{descer}",
            font_size=32,
            color=YELLOW,
        ).to_edge(UP, buff=3.5)

        lr_text = Text("Learning rate correto", font_size=26, color=GREEN).next_to(
            derivative_text, DOWN, buff=0.3
        )

        self.play(Write(derivative_text), run_time=1.5)
        self.wait(0.5)
        self.play(Write(lr_text), run_time=1)
        self.wait(1)

        # Gradient descent steps
        x_vals = [2.5, 1.8, 1.2, 0.7, 0.3, 0.1]

        for i, x_new in enumerate(x_vals[1:], 1):
            x_old = x_vals[i - 1]
            y_old = 0.5 * x_old**2 + 0.5
            y_new = 0.5 * x_new**2 + 0.5

            # Arrow showing step
            arrow = Arrow(
                self.axes.c2p(x_old, y_old),
                self.axes.c2p(x_new, y_new),
                color=ORANGE,
                stroke_width=3,
                buff=0.1,
            )

            new_point = Dot(self.axes.c2p(x_new, y_new), color=RED, radius=0.12)
            new_label = MathTex(f"w_{i}", font_size=28, color=RED).next_to(
                new_point, UP, buff=0.2
            )

            self.play(Create(arrow), run_time=0.8)
            self.wait(0.3)
            self.play(
                Transform(point, new_point),
                Transform(point_label, new_label),
                FadeOut(arrow),
                run_time=1,
            )
            self.wait(0.8)

        # Final message
        converged = Text("Convergiu!", font_size=32, color=GREEN, weight=BOLD).next_to(
            lr_text, DOWN, buff=0.3
        )

        self.play(Write(converged), run_time=1.5)
        self.wait(2)

        # Now show high learning rate problem
        self.play(
            FadeOut(point),
            FadeOut(point_label),
            FadeOut(converged),
            FadeOut(lr_text),
            FadeOut(derivative_text),
            run_time=1,
        )
        self.wait(0.5)

        self.show_overshooting()

    def show_overshooting(self):
        """Show what happens with too high learning rate"""

        # Warning text - MOVED TO TOP
        warning = Text(
            "Learning rate MUITO ALTO!", font_size=30, color=RED, weight=BOLD
        ).to_edge(UP, buff=3.5)
        self.play(Write(warning), run_time=1.5)
        self.wait(1)

        # Starting point
        x_start = 2
        point = Dot(
            self.axes.c2p(x_start, 0.5 * x_start**2 + 0.5),
            color=ORANGE,
            radius=0.12,
        )

        self.play(Create(point), run_time=1)
        self.wait(0.5)

        # Overshooting - bouncing back and forth WITH ROLLING ANIMATION
        x_vals = [2, -2.3, 2.5, -2.7, 2.9, -2.9]

        for i, x_new in enumerate(x_vals[1:], 1):
            x_old = x_vals[i - 1]
            y_old = 0.5 * x_old**2 + 0.5
            y_new = 0.5 * x_new**2 + 0.5

            arrow = Arrow(
                self.axes.c2p(x_old, y_old),
                self.axes.c2p(x_new, y_new),
                color=RED,
                stroke_width=3,
                buff=0.1,
            )

            # Create path for rolling animation
            num_steps = 15
            x_path = np.linspace(x_old, x_new, num_steps)

            self.play(Create(arrow), run_time=0.7)
            self.wait(0.2)

            # Animate point rolling along the curve
            for x_pos in x_path[1:]:
                y_pos = 0.5 * x_pos**2 + 0.5
                new_point = Dot(self.axes.c2p(x_pos, y_pos), color=ORANGE, radius=0.12)
                self.play(
                    Transform(point, new_point),
                    run_time=0.05,
                    rate_func=linear,
                )

            self.play(FadeOut(arrow), run_time=0.3)
            self.wait(0.3)

        diverged = Text(
            "Nunca converge!", font_size=32, color=RED, weight=BOLD
        ).next_to(warning, DOWN, buff=0.3)

        self.play(Write(diverged), run_time=1.5)
        self.wait(2.5)


class GradientDescent3D(Scene):
    """Gradient descent em 3D - superfície"""

    def construct(self):
        title = Text("Gradient Descent - 3D", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(2)

        # Show 3D surface representation
        self.show_3d_concept()
        self.wait(3)

    def show_3d_concept(self):
        """Show 3D surface concept with 2D projection"""

        # Create axes for top-down view
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        )

        labels = VGroup(
            MathTex("w_1", font_size=32).next_to(
                axes.x_axis.get_end(), RIGHT, buff=0.2
            ),
            MathTex("w_2", font_size=32).next_to(axes.y_axis.get_end(), UP, buff=0.2),
        )

        # Contour lines (representing height)
        contours = VGroup()
        for r in [0.3, 0.7, 1.1, 1.5, 1.9]:
            circle = Circle(
                radius=r * 1.5, color=BLUE, stroke_width=2, stroke_opacity=0.4
            )
            contours.add(circle)

        # Center (minimum)
        min_dot = Dot(ORIGIN, color=GREEN, radius=0.15)
        min_label = Text("Mínimo", font_size=28, color=GREEN).next_to(
            min_dot, DOWN, buff=0.4
        )

        # Label for contours
        contour_label = Text(
            "Curvas de nível\n(visão de cima)", font_size=24, slant=ITALIC
        ).to_corner(UL, buff=0.8)

        self.play(Create(axes), Write(labels), run_time=2)
        self.wait(1)
        self.play(Write(contour_label), run_time=1.5)
        self.wait(1)
        self.play(
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.3),
            run_time=2.5,
        )
        self.wait(1)
        self.play(Create(min_dot), Write(min_label), run_time=1.5)
        self.wait(1.5)

        # Starting point and descent path
        start_point = axes.c2p(1.5, 1.2)
        current_dot = Dot(start_point, color=RED, radius=0.12)

        self.play(Create(current_dot), run_time=1)
        self.wait(1)

        # Gradient descent path
        path_points = [
            (1.5, 1.2),
            (1.1, 0.85),
            (0.7, 0.5),
            (0.35, 0.2),
            (0.1, 0.05),
            (0, 0),
        ]

        gradient_text = (
            MathTex(
                r"\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{bmatrix}",
                font_size=32,
                color=YELLOW,
            )
            .to_edge(RIGHT, buff=0.8)
            .shift(UP * 1)
        )

        update_rule = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L",
            font_size=34,
            color=ORANGE,
        ).next_to(gradient_text, DOWN, buff=0.6)

        self.play(Write(gradient_text), run_time=2)
        self.wait(1.5)
        self.play(Write(update_rule), run_time=2)
        self.wait(1.5)

        # Animate descent
        for i in range(len(path_points) - 1):
            p1 = axes.c2p(*path_points[i])
            p2 = axes.c2p(*path_points[i + 1])

            arrow = Arrow(p1, p2, color=ORANGE, stroke_width=3, buff=0.1)
            new_dot = Dot(p2, color=RED, radius=0.12)

            self.play(Create(arrow), run_time=0.8)
            self.wait(0.3)
            self.play(Transform(current_dot, new_dot), FadeOut(arrow), run_time=1)
            self.wait(0.6)

        self.wait(2.5)


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

        # Step 2: Update w2 (going backwards) - NO RECTANGLE
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

        # Step 3: Update w1 (continue going backwards) - NO RECTANGLE
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


# Remove XORTraining scene
# class XORTraining removed as per user request


# ==============================================================================
# DURAÇÃO APROXIMADA:
# 1. NeuralNetworkArchitecture: ~45-50s
# 2. LossFunction: ~40-45s
# 3. GradientDescent2D: ~60-70s (com overshooting)
# 4. GradientDescent3D: ~45-50s
# 5. ChainRule: ~45-50s
# 6. Backpropagation: ~50-55s
# TOTAL: ~5-6 minutos
# ==============================================================================
