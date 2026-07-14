#include <torch/torch.h>

#include <iostream>

// Funcion de activacion no lineal: h = sigma(z).
torch::Tensor sigma(const torch::Tensor& x) { return torch::sin(x); }

int main() {
  // Fija la secuencia aleatoria para obtener los mismos valores en cada
  // ejecucion realizada con esta version de LibTorch.
  torch::manual_seed(0);

  // Matrices de pesos. LibTorch usa la forma [salidas, entradas].
  const auto W0 = torch::randn({4, 3}); // 3 entradas -> 4 salidas
  const auto W1 = torch::randn({3, 4}); // 4 entradas -> 3 salidas
  const auto W2 = torch::randn({2, 3}); // 3 entradas -> 2 salidas

  // Vector de entrada de la red.
  const auto x0 = torch::tensor({1.0F, 2.0F, 3.0F});
  const auto h0 = x0;

  // Forward de la capa 1: [4,3] @ [3] -> [4].
  const auto z1 = torch::matmul(W0, h0);
  const auto h1 = sigma(z1);

  // Forward de la capa 2: [3,4] @ [4] -> [3].
  const auto z2 = torch::matmul(W1, h1);
  const auto h2 = sigma(z2);

  // Forward de la capa 3: [2,3] @ [3] -> [2].
  const auto z3 = torch::matmul(W2, h2);
  const auto h3 = sigma(z3);

  std::cout << "W0 [4,3]:\n" << W0 << "\n\n";
  std::cout << "W1 [3,4]:\n" << W1 << "\n\n";
  std::cout << "W2 [2,3]:\n" << W2 << "\n\n";

  std::cout << "**************** ENTRADA ****************\n";
  std::cout << "x0 = h0 = " << x0 << "\n\n";

  std::cout << "**************** CAPA 1 *****************\n";
  std::cout << "z1 = W0 @ h0 = " << z1 << "\n";
  std::cout << "h1 = sin(z1) = " << h1 << "\n\n";

  std::cout << "**************** CAPA 2 *****************\n";
  std::cout << "z2 = W1 @ h1 = " << z2 << "\n";
  std::cout << "h2 = sin(z2) = " << h2 << "\n\n";

  std::cout << "**************** CAPA 3 *****************\n";
  std::cout << "z3 = W2 @ h2 = " << z3 << "\n";
  std::cout << "h3 = sin(z3) = " << h3 << "\n\n";

  std::cout << "Salida de la red: " << h3 << '\n';
  std::cout << "Por ahora la salida no tiene un significado aprendido: "
               "los pesos son aleatorios y no hay entrenamiento.\n";

  return 0;
}
