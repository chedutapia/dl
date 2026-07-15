# prueba3

Implementación con LibTorch de una red `3 → 4 → 3 → 2` que incorpora una
conexión residual, también llamada *skip connection*, desde la entrada hasta
la salida de la segunda capa oculta.

El ejemplo utiliza el mismo problema de clasificación binaria de `prueba2`: la
clase es 1 cuando al menos dos de las tres entradas binarias están activas.
Esto permite comparar una red secuencial con una residual.

## Arquitectura

El recorrido principal es:

```text
x [batch,3]
    │
    ├──────────────────────────────┐
    ▼                              │
Linear(3,4) + ReLU                 │ conexión corta
    │ [batch,4]                    │
    ▼                              │
Linear(4,3) + ReLU                 │
    │ [batch,3]                    │
    ▼                              │
   h2 + x ◄────────────────────────┘
    │ [batch,3]
    ▼
Linear(3,2)
    │
    ▼
logits [batch,2]
```

Matemáticamente:

```text
h1 = ReLU(W0 x + b0)
h2 = ReLU(W1 h1 + b1)
r  = h2 + x
o  = W2 r + b2
```

Las matrices de pesos tienen las formas:

| Parámetro | Forma |
|---|---:|
| `W0` | `[4,3]` |
| `W1` | `[3,4]` |
| `W2` | `[2,3]` |

## Implementación de la conexión residual

El `forward` conserva la entrada original mientras calcula el camino
principal:

```cpp
torch::Tensor forward(const torch::Tensor& x) {
  const auto h1 = torch::relu(capa1(x));
  const auto h2 = torch::relu(capa2(h1));
  const auto residual = h2 + x;
  return salida(residual);
}
```

La suma es elemento a elemento. Funciona directamente porque los dos tensores
tienen la misma forma:

```text
x  : [batch,3]
h2 : [batch,3]
```

Esto no es una concatenación. Una concatenación produciría `[batch,6]` y
requeriría una capa de salida `Linear(6,2)`. En esta implementación se utiliza
la interpretación residual de la figura: cada componente de entrada se suma a
la componente correspondiente de `h2`.

## Para qué sirve

Sin la conexión corta, el gradiente hacia la entrada del bloque debe atravesar
ambas capas ocultas. Si:

```text
r = h2(x) + x
```

entonces:

```text
∂r/∂x = ∂h2/∂x + I
```

El término identidad `I` proporciona una ruta directa para la información y
el gradiente. Las conexiones residuales pueden:

- Facilitar el entrenamiento de redes profundas.
- Reducir problemas de gradientes demasiado pequeños.
- Conservar información original de la entrada.
- Permitir que las capas aprendan una corrección o residuo respecto de `x`.

En vez de obligar al bloque a aprender una transformación completa `H(x)`, se
puede interpretar que aprende:

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

## Dimensiones diferentes

La suma directa exige que ambas ramas tengan la misma dimensión. Si `h2`
tuviera cinco características, sería necesaria una proyección aprendida:

```cpp
proyeccion(register_module("proyeccion", torch::nn::Linear(3, 5)));

auto residual = h2 + proyeccion(x);
```

## Entrenamiento

La red se entrena durante 1000 épocas con Adam, una tasa de aprendizaje de
`0.02` y entropía cruzada. La salida contiene dos logits, uno por clase;
`cross_entropy` aplica internamente la normalización necesaria durante el
entrenamiento.

Para evaluar se usa `softmax`, `argmax` y la exactitud sobre las ocho
combinaciones binarias. El modelo entrenado se guarda como
`modelo_prueba3.pt`.

## Compilar y ejecutar de forma independiente

Desde `prueba3`:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/../libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/prueba3
```

## Compilar desde la raíz

```bash
cmake --build build --target prueba3 -j
./build/prueba3/prueba3
```
