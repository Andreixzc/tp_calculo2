from manim import *
import numpy as np

DARK_BG = ManimColor("#05070d")
TEXT_PRIMARY = ManimColor("#f7f9ff")
TEXT_SECONDARY = ManimColor("#8b93a7")
NETWORK_EDGE = ManimColor("#4c556c")
NETWORK_GLOW = ManimColor("#58d0ff")
CURVE_COLOR = ManimColor("#ffa6ff")
POINT_COLOR = ManimColor("#58d0ff")
AXIS_COLOR = ManimColor("#2b2f3e")
GRID_COLOR = ManimColor("#3c4152")


class IntroAnimation(Scene):
    speed_scale = 2.0  # bigger -> slower (2x slower here)

    def play(self, *args, **kwargs):
        if "run_time" in kwargs:
            kwargs["run_time"] *= self.speed_scale
        else:
            kwargs["run_time"] = 1.0 * self.speed_scale
        super().play(*args, **kwargs)

    def wait(self, duration=1, *args, **kwargs):
        duration *= self.speed_scale
        return super().wait(duration, *args, **kwargs)

    def construct(self):
        self.camera.background_color = DARK_BG

        # titulo
        title = Text("Gradiente Descendente", font_size=70, weight=BOLD, color=TEXT_PRIMARY)

        self.play(FadeIn(title, shift=UP * 0.4), run_time=1.0)
        self.wait(0.6)
        self.play(
            title.animate.scale(0.6).to_edge(UP, buff=0.6),
            run_time=1.0,
        )

        # rede neural simples
        layer_sizes = [3, 4, 2]
        layers = VGroup()
        edges = VGroup()
        for layer_index, size in enumerate(layer_sizes):
            layer = VGroup(
                *[
                    Circle(radius=0.22, stroke_width=1.4, color=NETWORK_EDGE).set_fill(
                        NETWORK_EDGE, opacity=0.15
                    )
                    for _ in range(size)
                ]
            )
            layer.arrange(DOWN, buff=0.5)
            layer.shift(RIGHT * (layer_index - 1) * 2.2)
            layers.add(layer)

        for left_layer, right_layer in zip(layers[:-1], layers[1:]):
            for left_neuron in left_layer:
                for right_neuron in right_layer:
                    edges.add(
                        Line(
                            left_neuron.get_center(),
                            right_neuron.get_center(),
                            stroke_color=NETWORK_EDGE,
                            stroke_opacity=0.35,
                            stroke_width=1.2,
                        )
                    )

        network = VGroup(edges, layers).shift(UP * 0.5)

        self.play(LaggedStart(*[Create(edge) for edge in edges], lag_ratio=0.01), run_time=1.2)
        self.play(
            LaggedStart(
                *[FadeIn(neuron, scale=0.6) for layer in layers for neuron in layer],
                lag_ratio=0.05,
            ),
            run_time=1.0,
        )

        pulse_edges = edges.copy().set_stroke(color=NETWORK_GLOW, width=2.4, opacity=0.7)
        self.play(
            LaggedStart(
                *[
                    ShowPassingFlash(
                        edge,
                        time_width=0.4,
                        run_time=0.8,
                    )
                    for edge in pulse_edges
                ],
                lag_ratio=0.02,
            ),
            run_time=1.4,
        )

        self.play(FadeOut(network, shift=UP * 0.2), run_time=0.8)

        # grafico 1D com minimo local
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 4, 1],
            x_length=9,
            y_length=4.5,
            axis_config={"color": AXIS_COLOR, "stroke_width": 1.5},
        ).to_edge(DOWN, buff=0.7)
        axes.set_color_by_gradient(AXIS_COLOR, GRID_COLOR)

        def loss_curve(x):
            return 0.18 * x**4 - 0.9 * x**2 + 0.4 * np.sin(3 * x) + 1.6

        curve = axes.plot(loss_curve, x_range=[-3.6, 3.6], color=CURVE_COLOR, stroke_width=4.5)

        self.play(Create(axes), run_time=1.0)
        self.play(Create(curve), run_time=1.2)

        start_x = 2.8
        local_min_x = 1.1
        samples = 30
        x_values = np.linspace(start_x, local_min_x, samples)

        path_points = [axes.c2p(x, loss_curve(x)) for x in x_values]
        path = VMobject()
        path.set_points_as_corners(path_points)
        path.set_stroke(color=POINT_COLOR, width=2.5, opacity=0.8)

        descent_point = Dot(color=POINT_COLOR, radius=0.12).move_to(path_points[0])
        tracer = TracedPath(descent_point.get_center, stroke_color=POINT_COLOR, stroke_width=2)

        self.play(FadeIn(descent_point, scale=1.3))
        self.add(tracer)

        self.play(MoveAlongPath(descent_point, path), run_time=2.5, rate_func=rush_into)

        for _ in range(3):
            self.play(
                descent_point.animate.shift(UP * 0.08),
                run_time=0.15,
                rate_func=smooth
            )
            self.play(
                descent_point.animate.shift(DOWN * 0.08),
                run_time=0.15,
                rate_func=smooth
            )

        self.play(
            descent_point.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=0.6
        )

        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.0)
        self.wait(0.2)
