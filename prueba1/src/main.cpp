#include <torch/torch.h>

#include <iomanip>
#include <iostream>

struct XorNetImpl : torch::nn::Module {
  XorNetImpl()
      : hidden(register_module("hidden", torch::nn::Linear(2, 8))),
        output(register_module("output", torch::nn::Linear(8, 2))) {}

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(hidden(x));
    return output(x);
  }

  torch::nn::Linear hidden{nullptr};
  torch::nn::Linear output{nullptr};
};
TORCH_MODULE(XorNet);

int main() {
  torch::manual_seed(42);

  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                          : torch::kCPU);
  std::cout << "Dispositivo: " << device << '\n';

  auto inputs = torch::tensor({{0.0F, 0.0F},
                               {0.0F, 1.0F},
                               {1.0F, 0.0F},
                               {1.0F, 1.0F}})
                    .to(device);
  auto targets = torch::tensor({0, 1, 1, 0}, torch::kLong).to(device);

  XorNet model;
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.03));

  constexpr int epochs = 3000;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    model->train();
    optimizer.zero_grad();

    const auto logits = model->forward(inputs);
    const auto loss = torch::nn::functional::cross_entropy(logits, targets);
    loss.backward();
    optimizer.step();

    if (epoch == 1 || epoch % 100 == 0) {
      std::cout << "Epoch " << std::setw(4) << epoch
                << " | loss: " << std::fixed << std::setprecision(6)
                << loss.item<double>() << '\n';
    }
  }

  model->eval();
  torch::NoGradGuard no_grad;
  const auto probabilities = torch::softmax(model->forward(inputs), 1);
  const auto predictions = probabilities.argmax(1);

  std::cout << "\nPredicciones XOR:\n";
  for (int64_t i = 0; i < inputs.size(0); ++i) {
    std::cout << static_cast<int>(inputs[i][0].item<float>()) << " XOR "
              << static_cast<int>(inputs[i][1].item<float>()) << " = "
              << predictions[i].item<int64_t>() << " (confianza: "
              << std::setprecision(4)
              << probabilities[i][predictions[i]].item<float>() << ")\n";
  }

  torch::save(model, "xor_model.pt");
  std::cout << "\nModelo guardado en xor_model.pt\n";

  auto pars = model->parameters();
  for( const auto &p : pars )
    std::cout << p << std::endl << std::endl;

  return 0;
}
