#!/bin/bash

# Script to render all animations in high quality
cd /home/andrei/projects/tp_calc
source myenv/bin/activate

echo "🎬 Starting to render all animations in high quality (1080p60)..."
echo ""

# Render each animation file with -a flag (all scenes)
echo "✅ architecture_animation.py already rendered"

echo "📹 Rendering backpropagation_animation.py..."
manim -qh -a backpropagation_animation.py

echo "📹 Rendering chain_rule_animation.py..."
manim -qh -a chain_rule_animation.py

echo "📹 Rendering gradient_descent_2d_animation.py..."
manim -qh -a gradient_descent_2d_animation.py

echo "📹 Rendering gradient_descent_3d_animation.py..."
manim -qh -a gradient_descent_3d_animation.py

echo "📹 Rendering loss_function_animation.py..."
manim -qh -a loss_function_animation.py

echo "📹 Rendering perceptron_animation.py..."
manim -qh -a perceptron_animation.py

echo ""
echo "✅ All animations rendered successfully!"
echo "📁 Videos saved in: media/videos/"
