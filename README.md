# 🧠 Optimizers for Machine Learning  
### *A Mathematical, Visual, and Conceptual Exploration*

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*OE4zIhX9IhyQ9iE6Z0pO9A.gif" width="600" alt="Optimization Landscape">
</p>

---

## 📘 Overview

This repository provides an **in-depth, postgraduate-level exploration of optimization algorithms** used in machine learning — from **classical gradient-based methods** to **state-of-the-art curvature-aware optimizers**.

Each notebook blends:
- 🎓 **Rigorous mathematical foundations**
- 📈 **Interactive visualizations**
- 🧩 **Geometric intuitions**
- 💡 **Comparative experiments**

All explanations are self-contained and progressively build intuition for the underlying geometry, convergence behavior, and practical trade-offs of each method.

---

## 🧭 Repository Structure

```
optimizers-ml/
│
├── 01_basics/                                     # 🧩 Foundational Optimization Methods
│   ├── gradient_descent.ipynb                     # Standard Gradient Descent (GD) and Stochastic Gradient Descent (SGD).
│   ├── momentum_and_nesterov.ipynb                # Momentum-based acceleration: Classical Momentum and Nesterov Accelerated Gradient (NAG).
│   ├── adaptive_methods_adagrad_adam.ipynb        # Adaptive learning rate methods: AdaGrad, RMSprop, and Adam.
│
├── 02_sota_optimizers/                            # 🚀 State-of-the-Art (SOTA) Optimizers
│   ├── adamw_and_decoupled_weight_decay.ipynb     # AdamW and the principle of decoupled weight decay.
│   ├── ranger_lookahead_radam.ipynb               # RAdam (Rectified Adam), Lookahead, and Ranger.
│   ├── lion_and_adamx.ipynb                       # Modern, high-performance optimizers like Lion and AdamX.
│
├── 03_curvature_aware_optimizers-1/               # 🔷 Curvature-Aware (Part I): Foundations of Second-Order Methods
│   ├── natural_gradient_descent.ipynb             # Theory and implementation of Natural Gradient Descent (NGD).
│   ├── kfac_from_scratch.ipynb                    # Scalable curvature methods: Kronecker-Factored Approximate Curvature (K-FAC).
│   ├── riemannian_optimization.ipynb              # Optimization on manifolds: geodesic vs straight-line movement.
│
├── 04_curvature_aware_optimizers-2/               # 🔶 Curvature-Aware (Part II): Advanced and Practical Methods
│   ├── newton_and_levenberg_marquardt.ipynb       # Second-order curvature: Newton’s method, trust regions, and damping (Levenberg–Marquardt).
│   ├── quasi_newton_bfgs.ipynb                    # Quasi-Newton methods: BFGS and L-BFGS curvature approximations.
│   ├── ekfac_and_scaled_damping.ipynb             # Eigenvalue-Corrected K-FAC and improved damping techniques.
│   ├── hessian_spectrum_visualization.ipynb       # Hessian spectrum visualization: curvature, flat minima, and generalization.
│
├── 05_visualizations/                             # 🎨 Tools and Analysis for Optimization Landscapes
│   ├── optimization_landscapes.ipynb              # 2D and 3D loss landscape generation.
│   ├── loss_surface_geometry.ipynb                # Analyzing curvature: Hessian eigenvalues and condition numbers.
│   ├── convergence_animations.ipynb               # Animated visualizations of optimizer convergence trajectories.
│
├── assets/                                        # 🖼 Supporting media for notebooks
│   ├── figures/                                   # Static images and diagrams used in notebooks.
│   ├── gifs/                                      # Animated convergence visualizations and geodesic comparisons.
│
├── requirements.txt                               # 📦 List of Python dependencies (NumPy, Matplotlib, ipywidgets, etc.)
└── README.md                                      # 🧾 Main project overview and structure explanation.
```

---

## 🔬 Topics Covered

| Module | Theme | Highlights |
|--------|--------|-------------|
| **01. Gradient-Based Foundations** | Understanding how optimization connects calculus, linear algebra, and geometry. | Gradient Descent, Momentum, Nesterov Accelerated Gradient |
| **02. State-of-the-Art Optimizers** | Modern optimizers that dominate deep learning pipelines. | Adam, AdamW, RAdam, Lookahead, LION |
| **03. Curvature-Aware Methods** | Second-order and quasi-second-order optimizers using curvature information. | Natural Gradient, K-FAC, Shampoo, AdaHessian |
| **04. Visualization & Geometry** | Visual intuition of loss surfaces and optimizer dynamics. | 2D/3D plots, contour visualizations, optimization path animations |

---

## 🧮 Mathematical Depth

Each notebook includes:
- Derivations of update rules  
- Convergence analysis sketches  
- Theoretical connections to optimization theory and information geometry  
- Practical considerations: bias correction, numerical stability, and scaling  

Example snippet from *Natural Gradient Descent*:

```math
θ_{t+1} = θ_t - η F^{-1}(θ_t) ∇_θ L(θ_t)
```

where F(θ) is the Fisher Information Matrix, connecting optimization to information geometry.

---

## 🎨 Visualization Examples

- Optimization Trajectories over non-convex surfaces
- Learning rate schedules and their effects
- Curvature fields and metric tensors
- Animated convergence paths

<p align="center"> <img src="https://raw.githubusercontent.com/username/optimizers-ml/assets/figures/optimizer_trajectories.gif" width="600" alt="Optimizer Trajectories"> </p>

---

## 🧠 Learning Outcomes

- After working through this repository, you will:
- Understand the mathematical principles of modern optimizers
- Gain geometric intuition about curvature, conditioning, and step-size adaptation
- Appreciate the trade-offs between efficiency and generalization
- Be able to prototype and visualize optimizers in PyTorch / JAX

---

## 🧰 Tech Stack

- Python 3.11+
- Jupyter Notebooks
- NumPy, Matplotlib, Plotly, SymPy
- PyTorch or JAX for experiments
- Manim or matplotlib.animation for geometric visualizations

---

## Install dependencies:

```python
pip install -r requirements.txt
```

---

## 📊 Example Visualization

### Visualize optimizer trajectories over a 3D loss surface

```python
from visualization import plot_optimizer_path
plot_optimizer_path(loss_fn="rosenbrock", optimizers=["GD", "Adam", "NGD"])
```

---

## 🧩 Suggested Reading
- Goodfellow, Bengio, Courville (2016): Deep Learning — Chapter 8
- Martens & Grosse (2015): Optimizing Neural Networks with Kronecker-Factored Approximate Curvature
- Kingma & Ba (2015): Adam: A Method for Stochastic Optimization
- Zhang et al. (2022): The LION Optimizer
- Pascanu & Bengio (2014): Revisiting Natural Gradient Methods

---

## 🚀 Roadmap
 - Foundations and classical methods
 - SOTA adaptive optimizers
 - Curvature-aware optimizers (in progress)
 - Interactive dashboard (Streamlit) for optimizer comparison
 - Paper summaries and geometric notes

---

## 🧭 Contributing

Contributions are welcome!
If you have new visualizations, mathematical insights, or corrections:
Fork the repository
Create a new branch
Submit a pull request with a clear description

---

## 🧑‍🏫 Author

Agasthya 

Researcher in Optimization Theory

---

## 🧾 License
This repository is licensed under the MIT License.
Feel free to use, modify, and cite for academic and educational purposes.

---

## 🌟 Acknowledgements

Special thanks to the open-source and academic communities advancing our understanding of optimization theory and practice.

**“Optimization is not merely about descent — it’s about navigating the geometry of intelligence.”**