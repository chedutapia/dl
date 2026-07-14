# prueba2

Implementación en LibTorch de una red neuronal densa con arquitectura
`3 → 4 → 3 → 2`:

- Capa de entrada: 3 valores.
- Primera capa oculta: 4 neuronas con ReLU.
- Segunda capa oculta: 3 neuronas con ReLU.
- Capa de salida: 2 logits para clasificación.
- Matrices de pesos: `W0 [4,3]`, `W1 [3,4]` y `W2 [2,3]`.

El programa entrena la red con las ocho entradas binarias posibles. La clase
objetivo es 1 cuando al menos dos de las tres entradas valen 1.

## Compilar y ejecutar

LibTorch se comparte desde el directorio `../libtorch` del workspace:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/../libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/prueba2
```

La ejecución genera `modelo_prueba2.pt` en el directorio actual.

## La clase `RedImpl`

```cpp
struct RedImpl : torch::nn::Module {
```

`RedImpl` contiene la implementación real de la red y hereda de
`torch::nn::Module`, la clase base de LibTorch para componentes entrenables.
Esta herencia permite:

- Registrar las capas internas y descubrir sus parámetros.
- Mover toda la red entre CPU y GPU mediante `to(device)`.
- Cambiar entre los modos de entrenamiento y evaluación con `train()` y
  `eval()`.
- Obtener los pesos con `parameters()` para entregarlos a un optimizador.
- Guardar y cargar el modelo.

El sufijo `Impl` es una convención de la API C++ de LibTorch: `RedImpl`
contiene la implementación, mientras que `Red` será el contenedor público que
permite administrarla cómodamente.

### El constructor y las capas

```cpp
RedImpl()
    : capa1(register_module("capa1", torch::nn::Linear(3, 4))),
      capa2(register_module("capa2", torch::nn::Linear(4, 3))),
      salida(register_module("salida", torch::nn::Linear(3, 2))) {}
```

La lista de inicialización construye tres capas densas:

1. `Linear(3, 4)` recibe tres valores y produce cuatro.
2. `Linear(4, 3)` recibe cuatro valores y produce tres.
3. `Linear(3, 2)` recibe tres valores y produce los dos logits de salida.

Cada capa lineal calcula:

```text
y = x Wᵀ + b
```

LibTorch almacena una matriz de pesos con la forma `[salidas, entradas]`. Por
eso las tres matrices son:

| Parámetro | Capa | Forma |
|---|---|---|
| `W0` | `Linear(3, 4)` | `[4, 3]` |
| `W1` | `Linear(4, 3)` | `[3, 4]` |
| `W2` | `Linear(3, 2)` | `[2, 3]` |

Cada capa también posee un vector de sesgos: `[4]`, `[3]` y `[2]`,
respectivamente.

### Por qué se usa `register_module`

```cpp
register_module("capa1", torch::nn::Linear(3, 4))
```

Guardar una capa como atributo de C++ no basta. `register_module` informa al
módulo principal que esa capa forma parte del modelo. De esa manera LibTorch
puede:

- Incluir sus pesos y sesgos en `red->parameters()`.
- Calcular y almacenar sus gradientes.
- Mover sus tensores al dispositivo seleccionado.
- Guardar y recuperar sus parámetros.

Los nombres del registro forman una jerarquía como esta:

```text
capa1.weight
capa1.bias
capa2.weight
capa2.bias
salida.weight
salida.bias
```

### El método `forward`

```cpp
torch::Tensor forward(torch::Tensor x) {
  x = torch::relu(capa1(x));
  x = torch::relu(capa2(x));
  return salida(x);
}
```

`forward` define el recorrido de los datos. Para un lote de ocho ejemplos, las
formas evolucionan de esta manera:

```text
x [8,3]
   │
   ▼
Linear(3,4) + ReLU
   │ [8,4]
   ▼
Linear(4,3) + ReLU
   │ [8,3]
   ▼
Linear(3,2)
   │
   ▼
logits [8,2]
```

ReLU aplica `max(0, x)` elemento a elemento e introduce no linealidad. Sin
funciones de activación, varias capas lineales consecutivas serían
equivalentes a una única transformación lineal.

La última capa no aplica ReLU ni `softmax`: devuelve logits. Durante el
entrenamiento, `cross_entropy` recibe directamente esos logits y aplica
internamente la operación de normalización necesaria de una manera
numéricamente estable. Para presentar probabilidades después del
entrenamiento sí se utiliza `softmax`.

### Los atributos `Linear`

```cpp
torch::nn::Linear capa1{nullptr};
torch::nn::Linear capa2{nullptr};
torch::nn::Linear salida{nullptr};
```

`torch::nn::Linear` es un `ModuleHolder`: un contenedor que administra la
implementación de la capa, conceptualmente parecido a un puntero inteligente.
Los atributos comienzan en `nullptr` y reciben sus capas reales en la lista de
inicialización del constructor.

El contenedor permite escribir:

```cpp
capa1(x)
```

en lugar de invocar explícitamente `capa1->forward(x)`.

## Qué hace `TORCH_MODULE(Red)`

Después de la implementación se declara:

```cpp
TORCH_MODULE(Red);
```

Esta macro crea un `ModuleHolder` llamado `Red` que administra una instancia
de `RedImpl`. Conceptualmente es similar a:

```cpp
using Red = torch::nn::ModuleHolder<RedImpl>;
```

La macro permite construir el modelo de forma sencilla:

```cpp
Red red;
```

`red` es el contenedor y `red->` da acceso al objeto `RedImpl` que contiene:

```cpp
red->to(device);
red->train();
red->eval();
red->parameters();
red->forward(entradas);
```

La relación entre ambos tipos es:

```text
Red
└── ModuleHolder<RedImpl>
    └── RedImpl
        ├── capa1: Linear(3,4)
        ├── capa2: Linear(4,3)
        ├── salida: Linear(3,2)
        └── forward(x)
```

En resumen, `RedImpl` define la arquitectura y el cálculo de la red, mientras
que `TORCH_MODULE(Red)` crea el contenedor con el que se instancia, mueve,
entrena, guarda y utiliza esa implementación.

## Explicación de `main`

La función `main` ejecuta el ciclo completo del ejemplo:

```text
Configurar reproducibilidad
        ↓
Elegir CPU o GPU
        ↓
Crear datos y etiquetas
        ↓
Construir la red y el optimizador
        ↓
Entrenar durante 1000 épocas
        ↓
Evaluar las predicciones
        ↓
Guardar el modelo
```

### Semilla aleatoria

```cpp
torch::manual_seed(42);
```

LibTorch inicializa aleatoriamente los pesos y sesgos. Fijar la semilla hace
que esa inicialización sea reproducible: ejecuciones bajo las mismas
condiciones comienzan con los mismos parámetros. El número `42` es arbitrario
y puede reemplazarse por cualquier entero.

### Selección del dispositivo

```cpp
const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                        : torch::kCPU);
```

`torch::cuda::is_available()` comprueba si CUDA está disponible. El operador
ternario selecciona la GPU cuando es posible y la CPU en caso contrario. El
dispositivo elegido se conserva en una constante y se muestra en pantalla:

```cpp
std::cout << "Dispositivo: " << device << '\n';
```

### Entradas

```cpp
auto entradas = torch::tensor({{0.0F, 0.0F, 0.0F},
                                {0.0F, 0.0F, 1.0F},
                                {0.0F, 1.0F, 0.0F},
                                {0.0F, 1.0F, 1.0F},
                                {1.0F, 0.0F, 0.0F},
                                {1.0F, 0.0F, 1.0F},
                                {1.0F, 1.0F, 0.0F},
                                {1.0F, 1.0F, 1.0F}})
                     .to(device);
```

El tensor contiene las ocho combinaciones posibles de tres valores binarios.
Su forma es `[8,3]`: ocho ejemplos y tres características por ejemplo. Los
valores llevan el sufijo `F` porque las capas lineales trabajan con números de
punto flotante.

`.to(device)` coloca el tensor en el dispositivo seleccionado. El modelo, las
entradas y los objetivos siempre deben estar en el mismo dispositivo.

### Objetivos

```cpp
auto objetivos =
    torch::tensor({0, 0, 0, 1, 0, 1, 1, 1}, torch::kLong).to(device);
```

Cada posición contiene la clase correcta de la entrada correspondiente. La
clase 1 significa que al menos dos entradas valen 1:

| Entrada | Objetivo |
|---|---:|
| `[0,0,0]` | 0 |
| `[0,0,1]` | 0 |
| `[0,1,0]` | 0 |
| `[0,1,1]` | 1 |
| `[1,0,0]` | 0 |
| `[1,0,1]` | 1 |
| `[1,1,0]` | 1 |
| `[1,1,1]` | 1 |

El tipo `torch::kLong` representa enteros de 64 bits. `cross_entropy` espera
los objetivos en este formato porque cada valor es un índice de clase.

### Construcción del modelo

```cpp
Red red;
red->to(device);
mostrar_formas(red);
```

`Red red` construye el `ModuleHolder` generado por `TORCH_MODULE(Red)` y, por
lo tanto, una instancia de `RedImpl`. `red->to(device)` mueve todos los pesos y
sesgos registrados al dispositivo elegido.

`mostrar_formas(red)` no modifica el modelo; solamente muestra la arquitectura
y verifica las dimensiones de las matrices:

```text
W0: [4, 3]
W1: [3, 4]
W2: [2, 3]
```

### Optimizador Adam

```cpp
torch::optim::Adam optimizador(red->parameters(),
                               torch::optim::AdamOptions(0.02));
```

El optimizador recibe todos los parámetros entrenables registrados en la red.
Adam usa sus gradientes para actualizar pesos y sesgos buscando reducir la
pérdida. `0.02` es la tasa de aprendizaje: determina el tamaño aproximado de
las actualizaciones.

Una tasa demasiado pequeña puede hacer lento el aprendizaje; una demasiado
grande puede volverlo inestable.

### Cantidad de épocas

```cpp
constexpr int epocas = 1000;
```

Una época es una pasada completa por los datos de entrenamiento. Como las ocho
entradas se procesan juntas, cada iteración del ciclo utiliza el conjunto
completo. `constexpr` indica que el valor es una constante conocida durante la
compilación.

### Ciclo de entrenamiento

```cpp
for (int epoca = 1; epoca <= epocas; ++epoca) {
```

El cuerpo se repite 1000 veces. Cada repetición realiza cinco pasos centrales:

```text
Limpiar gradientes → predecir → medir error → calcular gradientes → actualizar
```

#### 1. Modo entrenamiento

```cpp
red->train();
```

Activa el modo de entrenamiento. Las capas de este ejemplo no cambian su
comportamiento, pero capas como `Dropout` y `BatchNorm` sí lo hacen.

#### 2. Limpieza de gradientes

```cpp
optimizador.zero_grad();
```

Los gradientes se acumulan en LibTorch. Esta operación elimina los valores de
la época anterior antes de calcular los nuevos.

#### 3. Propagación hacia adelante

```cpp
const auto logits = red->forward(entradas);
```

Las ocho entradas atraviesan las tres capas:

```text
[8,3] → Linear(3,4) + ReLU → [8,4]
      → Linear(4,3) + ReLU → [8,3]
      → Linear(3,2)        → [8,2]
```

El resultado tiene una fila por ejemplo y una puntuación, o logit, por clase.

#### 4. Cálculo de la pérdida

```cpp
const auto perdida =
    torch::nn::functional::cross_entropy(logits, objetivos);
```

La entropía cruzada compara los logits `[8,2]` con las clases correctas `[8]`
y devuelve un tensor escalar. Cuanto menor sea ese valor, mejor coinciden las
predicciones con los objetivos.

No se aplica `softmax` antes: `cross_entropy` trabaja directamente con logits
y realiza internamente la normalización de manera numéricamente estable.

#### 5. Retropropagación

```cpp
perdida.backward();
```

LibTorch recorre automáticamente el grafo de operaciones en sentido inverso y
calcula la derivada de la pérdida respecto de cada peso y sesgo:

```text
∂L/∂W0, ∂L/∂b0, ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2
```

Los gradientes resultantes quedan almacenados en los parámetros.

#### 6. Actualización de parámetros

```cpp
optimizador.step();
```

Adam utiliza los gradientes para modificar los parámetros. De manera
simplificada, el descenso de gradiente busca realizar:

```text
W ← W - tasa_de_aprendizaje × gradiente
```

Adam incorpora promedios adaptativos, pero conserva el mismo objetivo: reducir
la pérdida.

#### 7. Presentación del progreso

```cpp
if (epoca == 1 || epoca % 100 == 0) {
```

El programa muestra la primera época y luego cada 100 épocas. `std::setw`,
`std::fixed` y `std::setprecision` solamente controlan el formato. Como
`perdida` es un tensor escalar, `item<double>()` extrae su valor como un número
normal de C++.

### Evaluación

```cpp
red->eval();
torch::NoGradGuard sin_gradientes;
```

`eval()` cambia el modelo al modo de inferencia. `NoGradGuard` desactiva
temporalmente el seguimiento automático de gradientes, ya que durante la
evaluación no se ejecutará `backward()`. Esto reduce el uso de memoria y el
costo computacional.

### Probabilidades y predicciones

```cpp
const auto probabilidades = torch::softmax(red->forward(entradas), 1);
const auto predicciones = probabilidades.argmax(1);
```

`softmax` transforma los dos logits de cada ejemplo en probabilidades entre 0
y 1 que suman 1. El argumento `1` indica que debe operar sobre la dimensión de
las clases:

```text
dimensión 0: ejemplos
dimensión 1: clases
```

`argmax(1)` selecciona el índice de la probabilidad más alta en cada fila. El
resultado es un tensor `[8]` con una clase predicha por ejemplo.

### Exactitud

```cpp
const auto exactitud =
    predicciones.eq(objetivos).to(torch::kFloat).mean();
```

Esta expresión:

1. Compara las predicciones con los objetivos mediante `eq`.
2. Convierte `true` y `false` en `1.0` y `0.0`.
3. Calcula el promedio.

Por ejemplo, tres aciertos de cuatro producirían `0.75`, equivalente al 75 %.
Para mostrar el porcentaje, el programa extrae el valor con `item<float>()` y
lo multiplica por 100.

### Impresión de resultados

El último ciclo recorre las ocho entradas:

```cpp
for (int64_t i = 0; i < entradas.size(0); ++i) {
```

`predicciones[i].item<int64_t>()` obtiene la clase predicha como un entero de
C++. La confianza se selecciona con `probabilidades[i][clase]`, es decir, la
probabilidad asignada a la clase elegida.

Una línea de salida tiene esta forma:

```text
[0, 1, 1] -> 1 (confianza: 1.0000)
```

### Guardado

```cpp
torch::save(red, "modelo_prueba2.pt");
```

Guarda los pesos y sesgos aprendidos en el directorio desde el que se ejecuta
el programa. El archivo puede cargarse posteriormente en una red con la misma
arquitectura:

```cpp
Red red;
torch::load(red, "modelo_prueba2.pt");
red->eval();
```

El núcleo de una época puede resumirse así:

```cpp
optimizador.zero_grad();                 // Limpiar gradientes
auto logits = red->forward(entradas);    // Predecir
auto perdida = cross_entropy(...);       // Medir el error
perdida.backward();                      // Calcular gradientes
optimizador.step();                      // Actualizar parámetros
```

## Por qué hay dos salidas para una clasificación binaria

La última capa de la red es:

```cpp
salida(register_module("salida", torch::nn::Linear(3, 2)))
```

El modelo no entrena una sola salida: entrena simultáneamente una puntuación
para cada una de las dos clases. Para cada entrada devuelve:

```text
[logit_clase_0, logit_clase_1]
```

Por ejemplo:

```text
[ 5.2, -3.1] → clase 0
[-2.8,  4.7] → clase 1
```

La salida mayor determina la clase preferida. Cuando el objetivo es la clase
1, `cross_entropy` impulsa al modelo a aumentar el logit 1 respecto del logit
0. Las dos neuronas reciben gradientes y sus parámetros se actualizan.

Softmax transforma dos logits `[z0,z1]` en probabilidades:

```text
p0 = exp(z0) / (exp(z0) + exp(z1))
p1 = exp(z1) / (exp(z0) + exp(z1))
```

Las probabilidades suman 1. Aunque se resuelve una sola tarea, se utiliza una
salida explícita por cada clase.

### ¿Se puede usar una única salida?

Sí. En una clasificación estrictamente binaria puede usarse:

```cpp
salida(register_module("salida", torch::nn::Linear(3, 1)))
```

El único logit `z` se convertiría en la probabilidad de la clase 1 mediante
sigmoid:

```text
p(clase 1) = sigmoid(z)
p(clase 0) = 1 - sigmoid(z)
```

La predicción puede obtenerse con un umbral:

```text
sigmoid(z) >= 0.5 → clase 1
sigmoid(z) <  0.5 → clase 0
```

En ese diseño también deben cambiar los objetivos y la función de pérdida:

```cpp
auto objetivos = torch::tensor({{0.0F},
                                 {0.0F},
                                 {0.0F},
                                 {1.0F},
                                 {0.0F},
                                 {1.0F},
                                 {1.0F},
                                 {1.0F}})
                      .to(device);

auto perdida =
    torch::nn::functional::binary_cross_entropy_with_logits(logits,
                                                             objetivos);
```

Para evaluar se usarían sigmoid y un umbral en lugar de softmax y `argmax`:

```cpp
auto probabilidades = torch::sigmoid(red->forward(entradas));
auto predicciones = probabilidades.ge(0.5).to(torch::kLong);
```

Las dos formulaciones están relacionadas. En el caso de dos salidas:

```text
p1 = sigmoid(z1 - z0)
```

Por eso un único logit puede representar esencialmente la diferencia entre
los dos logits. Dos salidas conservan la arquitectura de la figura, son
didácticas y se extienden naturalmente a más clases. Una salida requiere menos
parámetros y es suficiente si el problema siempre será binario.

La capa actual tiene ocho parámetros:

```text
3 × 2 pesos + 2 sesgos = 8
```

Una capa con una salida tendría cuatro:

```text
3 × 1 peso + 1 sesgo = 4
```

## Los logits de salida

Los logits son las puntuaciones sin normalizar producidas directamente por la
última capa, antes de convertirlas en probabilidades:

```cpp
return salida(x);
```

Un logit puede ser positivo, negativo, mayor que 1 o menor que -1. Los logits
no son probabilidades y no tienen que sumar 1.

### Cómo se calculan

Después de las capas ocultas, cada ejemplo está representado por tres
activaciones:

```text
h² = [h²1, h²2, h²3]
```

La última capa calcula:

```text
z = h² (W²)ᵀ + b²
```

Como existen dos neuronas de salida:

```text
z = [z0,z1]
```

Cada neurona tiene sus propios pesos y sesgo:

```text
z0 = w00·h²1 + w01·h²2 + w02·h²3 + b0
z1 = w10·h²1 + w11·h²2 + w12·h²3 + b1
```

`z0` es la puntuación de la clase 0 y `z1` la puntuación de la clase 1.

### De logits a probabilidades

Durante la evaluación se usa:

```cpp
auto probabilidades = torch::softmax(red->forward(entradas), 1);
```

Por ejemplo:

```text
Logits:         [ 2.0, 1.0]
Probabilidades: [0.7311, 0.2689]
Predicción:     clase 0

Logits:         [-1.0, 2.0]
Probabilidades: [0.0474, 0.9526]
Predicción:     clase 1
```

Softmax produce valores entre 0 y 1 que suman 1 y conserva cuál de los logits
es mayor.

### Importa la diferencia entre logits

Para decidir la clase importa la diferencia entre los logits, no sus valores
absolutos. Estos tres pares producen las mismas probabilidades:

```text
[ 2.0,  1.0]
[12.0, 11.0]
[-3.0, -4.0]
```

En todos los casos `z0 - z1 = 1`. Sumar la misma constante a todos los logits
no cambia softmax. Por eso un logit positivo no implica por sí solo que esa
clase sea elegida:

```text
[2.0, 5.0] → clase 1
```

Tampoco un logit negativo implica que la clase sea descartada:

```text
[-2.0, -5.0] → clase 0
```

### Separación y confianza

Una separación mayor concentra más la probabilidad en la salida ganadora:

| Logits | Softmax aproximado |
|---|---|
| `[1.0,1.0]` | `[0.5000,0.5000]` |
| `[2.0,1.0]` | `[0.7311,0.2689]` |
| `[4.0,1.0]` | `[0.9526,0.0474]` |
| `[10.0,1.0]` | `[0.9999,0.0001]` |

La probabilidad suele presentarse como confianza, pero no garantiza que la
predicción sea correcta ni que el modelo esté bien calibrado.

### Cómo los usa `cross_entropy`

Durante el entrenamiento se pasan los logits directamente a la pérdida:

```cpp
auto logits = red->forward(entradas);
auto perdida =
    torch::nn::functional::cross_entropy(logits, objetivos);
```

`cross_entropy` combina internamente una operación equivalente a
`log_softmax` con la penalización de la clase correcta, haciéndolo de forma
numéricamente estable.

Si el objetivo es la clase 1 y la red produce `[3.0,-1.0]`, la pérdida será
alta porque favorece la clase 0. Para reducirla, el entrenamiento intentará
aumentar el segundo logit respecto del primero.

### Por qué `forward` no aplica softmax

El patrón correcto para entrenar es:

```cpp
auto logits = red->forward(entradas);
auto perdida =
    torch::nn::functional::cross_entropy(logits, objetivos);
```

Y para presentar resultados:

```cpp
auto logits = red->forward(entradas);
auto probabilidades = torch::softmax(logits, 1);
```

Aplicar softmax dentro de `forward` y volver a pasar ese resultado a
`cross_entropy` sería una normalización adicional innecesaria y menos estable.

### Forma del tensor de logits

Las entradas tienen forma `[8,3]` y la salida tiene forma `[8,2]`:

```text
[
  [logit_0_ejemplo_0, logit_1_ejemplo_0],
  [logit_0_ejemplo_1, logit_1_ejemplo_1],
  ...
  [logit_0_ejemplo_7, logit_1_ejemplo_7]
]
```

En `softmax(logits, 1)`, el argumento `1` selecciona la dimensión de clases.
Después, `argmax(1)` escoge la columna con mayor valor en cada fila.

En resumen, los logits son las puntuaciones crudas de la última capa.
`cross_entropy` los utiliza directamente durante el entrenamiento, mientras
que `softmax` los convierte en probabilidades durante la evaluación.
