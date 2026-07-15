# backprop0

Ejemplo didáctico de entrenamiento y backpropagation manual con LibTorch. La
red aprende a aproximar una función de una variable a partir de mediciones con
ruido, sin utilizar `torch::nn::Module` ni un optimizador predefinido.

El objetivo es observar explícitamente las etapas que normalmente abstraen
`torch::nn` y `torch::optim`:

1. Crear parámetros entrenables.
2. Ejecutar el feedforward.
3. Calcular una función de pérdida.
4. Obtener gradientes mediante autograd.
5. Actualizar manualmente pesos y sesgos.
6. Borrar los gradientes antes de la siguiente época.

## Datos

La variable de entrada contiene 200 puntos uniformemente distribuidos entre
-3 y 3:

```cpp
const auto x = torch::linspace(-3.0F, 3.0F, 200).reshape({-1, 1});
```

El fenómeno que se quiere aproximar es:

```text
y_real = exp(-x²) sin(3x)
```

Las mediciones agregan ruido normal para simular error experimental:

```cpp
const auto y_real = torch::exp(-x.pow(2)) * torch::sin(3 * x);
const auto y = y_real + 0.15F * torch::randn_like(y_real);
```

Tanto `x` como `y` tienen forma `[200,1]`. La semilla se fija con
`torch::manual_seed(0)` para que la generación de pesos y ruido sea
reproducible con la misma versión de LibTorch.

## Arquitectura

La red es un perceptrón multicapa para regresión:

```text
Entrada            Capa oculta 1       Capa oculta 2       Salida
[200,1]        →   5 neuronas      →   5 neuronas      →   1 valor
                        sigmoid             sigmoid           lineal
```

Sus parámetros son tensores creados manualmente:

| Parámetro | Forma | Conexión |
|---|---:|---|
| `W0` | `[5,1]` | Entrada → capa oculta 1 |
| `b0` | `[5,1]` | Sesgo de capa oculta 1 |
| `W1` | `[5,5]` | Capa oculta 1 → capa oculta 2 |
| `b1` | `[5,1]` | Sesgo de capa oculta 2 |
| `W2` | `[1,5]` | Capa oculta 2 → salida |
| `b2` | `[1,1]` | Sesgo de salida |

Cada parámetro se crea con `requires_grad(true)`:

```cpp
auto W0 = torch::randn(
    {5, 1},
    torch::TensorOptions().dtype(torch::kFloat).requires_grad(true));
```

Esto indica a autograd que debe seguir las operaciones realizadas con el
tensor y calcular la derivada de la pérdida respecto de él.

## Feedforward

Todo el conjunto de 200 mediciones se procesa como un único batch. Primero se
transpone `x` de `[200,1]` a `[1,200]` para realizar las multiplicaciones:

```text
h0 = xᵀ                         [1,200]
z1 = W0 h0 + b0                [5,200]
h1 = sigmoid(z1)               [5,200]
z2 = W1 h1 + b1                [5,200]
h2 = sigmoid(z2)               [5,200]
z3 = W2 h2 + b2                [1,200]
y_pred = z3ᵀ                   [200,1]
```

Los sesgos `[5,1]` y `[1,1]` se expanden automáticamente mediante
broadcasting sobre las 200 columnas.

La salida es lineal porque se trata de regresión: el resultado debe poder
tomar valores reales positivos o negativos. Sigmoid solamente se utiliza en
las capas ocultas para introducir no linealidad.

## Función de pérdida

Se utiliza el error cuadrático medio o MSE:

```cpp
const auto loss = (y - y_pred).pow(2).mean();
```

Matemáticamente:

```text
MSE = (1/N) Σ (yᵢ - ŷᵢ)²
```

Los errores grandes reciben una penalización cuadrática. El entrenamiento
busca modificar los parámetros para minimizar este valor.

## Backpropagation con autograd

La instrucción:

```cpp
loss.backward();
```

recorre el grafo de operaciones en sentido inverso y calcula:

```text
∂Loss/∂W0, ∂Loss/∂b0,
∂Loss/∂W1, ∂Loss/∂b1,
∂Loss/∂W2, ∂Loss/∂b2
```

Los resultados quedan almacenados en `W0.grad()`, `b0.grad()` y los demás
parámetros. En este ejemplo, LibTorch calcula los gradientes pero la
actualización se implementa manualmente.

## Descenso de gradiente manual

La tasa de aprendizaje es `0.1` y se realizan 50.000 épocas:

```cpp
constexpr float learning_rate = 0.1F;
constexpr int epochs = 50000;
```

Cada parámetro se actualiza en la dirección opuesta a su gradiente:

```text
parámetro ← parámetro - learning_rate × gradiente
```

En LibTorch se utiliza una suma in-place con factor negativo:

```cpp
W0.add_(W0.grad(), -learning_rate);
```

Las actualizaciones se ejecutan dentro de:

```cpp
torch::NoGradGuard no_grad;
```

Esto evita que autograd agregue la propia actualización de parámetros al grafo
de derivadas. Los pesos deben continuar siendo tensores hoja entre épocas.

## Limpieza de gradientes

Autograd acumula los gradientes de sucesivas llamadas a `backward()`. Después
de cada actualización se borran explícitamente:

```cpp
W0.grad().zero_();
b0.grad().zero_();
```

La operación se repite para los seis parámetros. Si se omitiera este paso, el
gradiente de cada época se sumaría a los anteriores y las actualizaciones no
representarían el descenso de gradiente previsto.

## Predicción

Después del entrenamiento se repite el feedforward dentro de un
`torch::NoGradGuard`. Ya no se necesita construir un grafo porque no habrá otra
llamada a `backward()`.

El programa muestra la pérdida final y cinco muestras con el formato:

```text
[x, y medida, y predicha]
```

Con LibTorch 2.7.1 para CPU se obtuvo aproximadamente:

```text
Loss inicial = 4.33938
Loss final   = 0.022825
```

Los valores exactos pueden variar si cambia la versión de LibTorch, el
dispositivo o el generador de números aleatorios.

## Compilar y ejecutar de forma independiente

Desde `backprop0`:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/../libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/backprop0
```

## Compilar desde la raíz

Si el build general ya está configurado:

```bash
cmake --build build --target backprop0 -j
./build/backprop0/backprop0
```

Este ejemplo no genera un gráfico como la versión de Python. Su objetivo es
mostrar el entrenamiento y algunas predicciones en la consola; una
visualización puede incorporarse posteriormente exportando los datos o usando
una biblioteca gráfica para C++.
