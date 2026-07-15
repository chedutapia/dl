#include <torch/torch.h>

#include <iomanip>
#include <iostream>

torch::Tensor sigma(const torch::Tensor& z) { return torch::sigmoid(z); }

int main() {
  torch::manual_seed(0);

  // Variable que controlamos: 200 puntos entre -3 y 3, con forma [200, 1].
  const auto x = torch::linspace(-3.0F, 3.0F, 200).reshape({-1, 1});

  // Fenomeno fisico y mediciones con ruido experimental simulado.
  const auto y_real = torch::exp(-x.pow(2)) * torch::sin(3 * x);
  const auto y = y_real + 0.07F * torch::randn_like(y_real);

  // Arquitectura 1 -> 5 -> 5 -> 1. Estos tensores son hojas del grafo de
  // autograd y almacenaran sus gradientes despues de Loss.backward().
  auto W0 = torch::randn(
      {5, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
  auto b0 = torch::randn(
      {5, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));

  auto W1 = torch::randn(
      {5, 5}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
  auto b1 = torch::randn(
      {5, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));

  auto W2 = torch::randn(
      {1, 5}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
  auto b2 = torch::randn(
      {1, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));

  constexpr float learning_rate = 0.1F;
  constexpr int epochs = 50000;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Feedforward. Todo el conjunto de entrenamiento forma un unico batch.
    const auto h0 = x.transpose(0, 1);      // [1, 200]
    const auto z1 = torch::matmul(W0, h0) + b0; // [5, 200]
    const auto h1 = sigma(z1);                   // [5, 200]
    const auto z2 = torch::matmul(W1, h1) + b1; // [5, 200]
    const auto h2 = sigma(z2);                   // [5, 200]
    const auto z3 = torch::matmul(W2, h2) + b2; // [1, 200]
    const auto y_pred = z3.transpose(0, 1);      // [200, 1]

    // Error cuadratico medio.
    const auto loss = (y - y_pred).pow(2).mean();

    // Calcula y acumula los gradientes de todos los parametros hoja.
    loss.backward();

    // Actualiza los parametros sin registrar estas operaciones en autograd.
    {
      torch::NoGradGuard no_grad;
      W0.add_(W0.grad(), -learning_rate);
      b0.add_(b0.grad(), -learning_rate);
      W1.add_(W1.grad(), -learning_rate);
      b1.add_(b1.grad(), -learning_rate);
      W2.add_(W2.grad(), -learning_rate);
      b2.add_(b2.grad(), -learning_rate);
    }

    // Autograd acumula gradientes por defecto; hay que borrarlos antes de la
    // siguiente epoca.
    W0.grad().zero_();
    b0.grad().zero_();
    W1.grad().zero_();
    b1.grad().zero_();
    W2.grad().zero_();
    b2.grad().zero_();

    if (epoch % 500 == 0) {
      std::cout << "Epoch " << std::setw(5) << epoch
                << "   Loss = " << std::fixed << std::setprecision(5)
                << loss.item<float>() << '\n';
    }
  }

  // Predicciones luego del entrenamiento. No se necesita construir el grafo
  // porque ya no calcularemos gradientes.
  torch::Tensor y_pred;
  {
    torch::NoGradGuard no_grad;
    const auto h0 = x.transpose(0, 1);
    const auto z1 = torch::matmul(W0, h0) + b0;
    const auto h1 = sigma(z1);
    const auto z2 = torch::matmul(W1, h1) + b1;
    const auto h2 = sigma(z2);
    const auto z3 = torch::matmul(W2, h2) + b2;
    y_pred = z3.transpose(0, 1);
  }

  const auto final_loss = (y - y_pred).pow(2).mean().item<float>();
  std::cout << "\nLoss final = " << std::setprecision(6) << final_loss << '\n';
  std::cout << "Muestras [x, y medida, y predicha]:\n";
  for (const int index : {0, 50, 100, 150, 199}) {
    std::cout << std::setw(9) << x[index].item<float>() << "  "
              << std::setw(9) << y[index].item<float>() << "  "
              << std::setw(9) << y_pred[index].item<float>() << '\n';
  }

  return 0;
}
