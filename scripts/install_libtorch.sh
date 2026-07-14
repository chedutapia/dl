#!/usr/bin/env bash

set -euo pipefail

readonly VERSION="${1:-2.7.1}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly INSTALL_DIR="${WORKSPACE_DIR}/libtorch"
readonly ARCHIVE="${WORKSPACE_DIR}/libtorch-${VERSION}-cpu.zip"
readonly URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcpu.zip"

for command_name in curl unzip; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Error: falta el comando '${command_name}'." >&2
    exit 1
  fi
done

if [[ -e "${INSTALL_DIR}" ]]; then
  if [[ -f "${INSTALL_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    echo "LibTorch ya esta instalado en ${INSTALL_DIR}."
    exit 0
  fi

  echo "Error: ${INSTALL_DIR} existe, pero no parece una instalacion valida." >&2
  echo "Muevelo o eliminalo manualmente antes de continuar." >&2
  exit 1
fi

cleanup() {
  rm -f -- "${ARCHIVE}"
}
trap cleanup EXIT

echo "Descargando LibTorch ${VERSION} para CPU..."
curl --fail --location --progress-bar --output "${ARCHIVE}" "${URL}"

echo "Validando el archivo descargado..."
unzip -tq "${ARCHIVE}"

echo "Extrayendo LibTorch en ${WORKSPACE_DIR}..."
unzip -q "${ARCHIVE}" -d "${WORKSPACE_DIR}"

if [[ ! -f "${INSTALL_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "Error: la extraccion no produjo una instalacion valida." >&2
  exit 1
fi

echo "LibTorch ${VERSION} instalado correctamente en ${INSTALL_DIR}."
