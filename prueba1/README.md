# prueba1

Primer proyecto con LibTorch en C++. Entrena una red neuronal pequeña para
resolver XOR, muestra sus predicciones y guarda el modelo entrenado.

## Requisitos

- Linux o WSL2
- Compilador compatible con C++17
- CMake 3.16 o posterior
- LibTorch

## LibTorch compartido

Este workspace tiene LibTorch CPU instalado en el directorio hermano
`../libtorch`. Otros proyectos pueden reutilizar la misma instalación.

La distribución utilizada es **LibTorch 2.7.1 / CPU / cxx11 ABI**, obtenida
desde <https://download.pytorch.org/libtorch/cpu/>.

## Compilar

Desde `prueba1`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../libtorch"
cmake --build build -j
```

## Ejecutar

```bash
./build/prueba1
```

Al finalizar se genera `xor_model.pt` en el directorio desde el cual ejecutes
el programa.

## Estructura

```text
prueba1/
├── CMakeLists.txt
├── README.md
└── src/
    └── main.cpp
```
