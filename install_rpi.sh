#!/bin/bash
# Script de instalación rápida para Raspberry Pi 5
# Uso: chmod +x install_rpi.sh && ./install_rpi.sh

set -e  # Salir si hay error

echo "================================"
echo "Instalación - Concentration Monitor"
echo "Raspberry Pi 5"
echo "================================"

# Paso 1: Actualizar sistema
echo ""
echo "📦 Paso 1: Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

# Paso 2: Instalar dependencias
echo ""
echo "🔧 Paso 2: Instalando dependencias del sistema..."
sudo apt install -y \
  python3.11 \
  python3.11-dev \
  python3-pip \
  libatlas-base-dev \
  libjasper-dev \
  libtiff5 \
  libharfbuzz0b \
  libwebp6 \
  libjasper1 \
  libopenjp2-7 \
  libopenjp2-7-dev \
  libcamera0 \
  libcamera-apps \
  libcamera-tools \
  python3-libcamera \
  python3-kms++

# Paso 3: Crear entorno virtual
echo ""
echo "🐍 Paso 3: Creando entorno virtual..."
if [ ! -d "venv" ]; then
  python3.11 -m venv venv
  echo "✓ Entorno virtual creado"
else
  echo "✓ Entorno virtual ya existe"
fi

# Paso 4: Activar entorno y instalar dependencias
echo ""
echo "📥 Paso 4: Instalando paquetes Python..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Paso 5: Descargar modelo
echo ""
echo "🤖 Paso 5: Descargando modelo MediaPipe..."
python download_model.py

# Paso 6: Configurar Coral TPU (si es aplicable)
echo ""
echo "🦀 Paso 6: Coral TPU (opcional)..."
read -p "¿Tienes un Coral TPU conectado? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
  echo "Configurando acceso USB..."
  sudo usermod -a -G plugdev $USER
  echo "⚠️  Necesitas hacer logout y login para que los cambios de grupo surjan efecto"
  echo "O ejecuta: newgrp plugdev"
fi

# Paso 7: Crear carpetas
echo ""
echo "📁 Paso 7: Creando directorios..."
mkdir -p logs models
echo "✓ Directorios listos"

# Resumen
echo ""
echo "================================"
echo "✅ INSTALACIÓN COMPLETADA"
echo "================================"
echo ""
echo "Próximos pasos:"
echo "1. Conecta la cámara Raspberry Pi"
echo "2. Si agregaste grupos para Coral TPU, haz logout/login"
echo "3. Activa el entorno: source venv/bin/activate"
echo "4. Ejecuta: python main.py"
echo ""
echo "Para ayuda: python main.py --help"
echo ""
