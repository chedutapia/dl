#include <torch/torch.h>

#include <iomanip>
#include <iostream>

// Red 3 -> 4 -> 3 -> 2 con una conexion residual desde la entrada hasta la
// salida de la segunda capa oculta.
struct RedResidualImpl : torch::nn::Module {
  RedResidualImpl()
      : capa1(register_module("capa1", torch::nn::Linear(3, 4))),
        capa2(register_module("capa2", torch::nn::Linear(4, 3))),
        salida(register_module("salida", torch::nn::Linear(3, 2))) {}

  torch::Tensor forward(const torch::Tensor& x) {
    const auto h1 = torch::relu(capa1(x));
    const auto h2 = torch::relu(capa2(h1));
    const auto residual = h2 + x;
    return salida(residual); // logits para las dos clases
  }

  torch::nn::Linear capa1{nullptr};  // W0: [4,3]
  torch::nn::Linear capa2{nullptr};  // W1: [3,4]
  torch::nn::Linear salida{nullptr}; // W2: [2,3]
};
TORCH_MODULE(RedResidual);

void mostrar_arquitectura(const RedResidual& red) {
  std::cout << "Arquitectura principal: 3 -> 4 -> 3 -> 2\n"
            << "Conexion residual: x + h2\n"
            << "W0: " << red->capa1->weight.sizes() << '\n'
            << "W1: " << red->capa2->weight.sizes() << '\n'
            << "W2: " << red->salida->weight.sizes() << "\n\n";
}

int main() {
  torch::manual_seed(42);

  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                          : torch::kCPU);
  std::cout << "Dispositivo: " << device << '\n';

  // Las ocho combinaciones binarias de tres entradas. La clase 1 representa
  // que al menos dos de las tres entradas estan activas.
  auto entradas = torch::tensor({{0.0F, 0.0F, 0.0F},
                                  {0.0F, 0.0F, 1.0F},
                                  {0.0F, 1.0F, 0.0F},
                                  {0.0F, 1.0F, 1.0F},
                                  {1.0F, 0.0F, 0.0F},
                                  {1.0F, 0.0F, 1.0F},
                                  {1.0F, 1.0F, 0.0F},
                                  {1.0F, 1.0F, 1.0F}})
                       .to(device);
  auto objetivos = torch::tensor({0, 0, 0, 1, 0, 1, 1, 1}, torch::kLong)
                       .to(device);

  RedResidual red;
  red->to(device);
  mostrar_arquitectura(red);

  torch::optim::Adam optimizador(red->parameters(),
                                 torch::optim::AdamOptions(0.02));

  constexpr int epocas = 1000;
  for (int epoca = 1; epoca <= epocas; ++epoca) {
    red->train();
    optimizador.zero_grad();

    const auto logits = red->forward(entradas);
    const auto perdida =
        torch::nn::functional::cross_entropy(logits, objetivos);
    perdida.backward();
    optimizador.step();

    if (epoca == 1 || epoca % 100 == 0) {
      std::cout << "Epoca " << std::setw(4) << epoca
                << " | perdida: " << std::fixed << std::setprecision(6)
                << perdida.item<double>() << '\n';
    }
  }

  red->eval();
  torch::NoGradGuard sin_gradientes;
  const auto logits = red->forward(entradas);
  const auto probabilidades = torch::softmax(logits, 1);
  const auto predicciones = probabilidades.argmax(1);
  const auto exactitud = predicciones.eq(objetivos).to(torch::kFloat).mean();

  std::cout << "\nPredicciones (clase 1 = al menos dos unos):\n";
  for (int64_t i = 0; i < entradas.size(0); ++i) {
    const auto clase = predicciones[i].item<int64_t>();
    std::cout << '[' << entradas[i][0].item<int>() << ", "
              << entradas[i][1].item<int>() << ", "
              << entradas[i][2].item<int>() << "] -> " << clase
              << " (confianza: " << std::setprecision(4)
              << probabilidades[i][clase].item<float>() << ")\n";
  }

  std::cout << "Exactitud: " << exactitud.item<float>() * 100.0F << "%\n";
  torch::save(red, "modelo_prueba3.pt");
  std::cout << "Modelo guardado en modelo_prueba3.pt\n";
  return 0;
}
