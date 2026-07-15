# Redes neuronales en C++ con LibTorch

Workspace de ejemplos progresivos para aprender redes neuronales en C++ con
la API oficial de PyTorch, LibTorch.

## Estructura

```text
dl/
├── CMakeLists.txt         # Construcción conjunta de todos los ejemplos
├── libtorch/              # Dependencia local compartida; no se versiona
├── scripts/
│   └── install_libtorch.sh
├── prueba0/               # Operaciones elementales con tensores
├── prueba1/               # Red sencilla para XOR
├── prueba2/               # Red densa 3 → 4 → 3 → 2
├── prueba3/               # Red 3 → 4 → 3 → 2 con conexión residual
└── backprop0/             # Autograd y backpropagation manual
```

Cada proyecto tiene su propio `CMakeLists.txt`, código fuente y README. Todos
utilizan una única instalación de LibTorch ubicada en `./libtorch`.

## Requisitos

- Linux o WSL2.
- Compilador compatible con C++17.
- CMake 3.16 o posterior.
- `curl` y `unzip` para ejecutar el instalador.

Este workspace fue probado con Ubuntu 20.04 en WSL2, GCC 9.4, CMake 3.16 y
LibTorch 2.7.1 para CPU con cxx11 ABI.

## Instalar LibTorch

LibTorch no se guarda en Git porque contiene cientos de megabytes de headers y
binarios específicos de cada plataforma. El script instala una versión fija y
compartida en `./libtorch`:

```bash
./scripts/install_libtorch.sh
```

El instalador descarga desde el repositorio oficial de PyTorch la distribución
**LibTorch 2.7.1 / CPU / shared with dependencies / cxx11 ABI**. Comprueba el
archivo ZIP antes de extraerlo y no reemplaza una instalación existente.

La instalación fue exitosa si existe:

```text
libtorch/share/cmake/Torch/TorchConfig.cmake
```

La versión puede sobrescribirse mediante el primer argumento, siempre que el
archivo correspondiente exista en el servidor oficial:

```bash
./scripts/install_libtorch.sh 2.7.1
```

Las distribuciones CUDA requieren elegir una variante compatible con el driver
y no están cubiertas por este instalador para CPU.

## Compilar todos los proyectos

Desde la raíz del workspace:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Los ejecutables quedan agrupados por proyecto:

```text
build/prueba0/prueba0
build/prueba1/prueba1
build/prueba2/prueba2
build/prueba3/prueba3
build/backprop0/backprop0
```

Se puede construir y ejecutar solamente un ejemplo:

```bash
cmake --build build --target prueba0 -j
./build/prueba0/prueba0
```

Los objetivos disponibles son `prueba0`, `prueba1`, `prueba2`, `prueba3` y
`backprop0`.

## Compilar un proyecto de forma independiente

Desde el directorio del ejemplo, se indica a CMake dónde está la instalación
compartida:

```bash
cd prueba0
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/../libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/prueba0
```

Para los otros ejemplos se reemplaza `prueba0` por `prueba1`, `prueba2`,
`prueba3` o `backprop0`.

## Visual Studio Code

El repositorio incluye una configuración compartida en `.vscode/` para la
extensión C/C++ de Microsoft. Define:

- `/usr/bin/g++` como compilador.
- C++17 como estándar.
- IntelliSense para GCC en Linux/WSL.
- Los dos directorios de headers requeridos por LibTorch.

La configuración supone que LibTorch está en `${workspaceFolder}/libtorch`.
Después de instalarlo, si IntelliSense todavía muestra errores:

1. Abrir la paleta con `Ctrl+Shift+P`.
2. Ejecutar `C/C++: Reset IntelliSense Database`.
3. Ejecutar `Developer: Reload Window`.

## Proyectos

- [`prueba0`](prueba0/README.md): creación de tensores, productos
  matriz-vector y forward manual.
- [`prueba1`](prueba1/README.md): entrenamiento de un MLP para aprender XOR.
- [`prueba2`](prueba2/README.md): arquitectura `3 → 4 → 3 → 2`, entrenamiento,
  logits, probabilidades y documentación detallada.
- [`prueba3`](prueba3/README.md): variante `3 → 4 → 3 → 2` con una conexión
  residual desde la entrada hasta la segunda capa oculta.
- [`backprop0`](backprop0/README.md): ajuste de una función con una red
  `1 → 5 → 5 → 1`, autograd y descenso de gradiente implementado manualmente.

## Archivos ignorados

Git excluye deliberadamente:

- La instalación `libtorch/` y sus ZIP.
- Los directorios `build/` de CMake.
- Los modelos entrenados `*.pt`.
- Configuraciones personales de VS Code diferentes de las dos compartidas.
