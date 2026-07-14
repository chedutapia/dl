# prueba0

Ejemplo elemental de tensores con LibTorch equivalente al código introductorio
en Python. Construye manualmente una red feedforward `3 → 4 → 3 → 2`, sin
usar `torch::nn::Module` y sin entrenarla.

El programa muestra:

- Creación de tensores con `torch::tensor` y `torch::randn`.
- Formas de las matrices de pesos.
- Multiplicación matriz-vector con `torch::matmul`.
- Aplicación elemento a elemento de `torch::sin`.
- Valores intermedios `z1`, `h1`, `z2`, `h2`, `z3` y `h3`.

El forward implementado es:

```text
h0 = x0
z1 = W0 @ h0    h1 = sin(z1)
z2 = W1 @ h1    h2 = sin(z2)
z3 = W2 @ h2    h3 = sin(z3)
```

Los pesos se generan aleatoriamente y no se optimizan, por lo que `h3` todavía
no representa una predicción aprendida.

## Compilar y ejecutar

Desde `prueba0`:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/../libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/prueba0
```
