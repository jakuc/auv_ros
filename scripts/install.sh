#!/bin/bash
# install.sh - konfiguracja nowego sprzętu dla auv_ros
# Uruchom jako root: sudo ./scripts/install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_USER="${SUDO_USER:-$USER}"

if [ "$EUID" -ne 0 ]; then
    echo "Uruchom jako root: sudo $0"
    exit 1
fi

echo "========================================"
echo "  auv_ros - instalacja środowiska"
echo "========================================"
echo "Repozytorium: $REPO_DIR"
echo "Użytkownik:   $TARGET_USER"
echo ""

# --- Krok 1: Docker ---
if ! command -v docker &> /dev/null; then
    echo "[1/3] Instalacja Docker..."
    apt-get update -qq
    apt-get install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    usermod -aG docker "$TARGET_USER"
    echo "    Docker zainstalowany. Wymagane wylogowanie/zalogowanie dla uprawnień."
else
    echo "[1/3] Docker już zainstalowany - pomijam."
fi

# --- Krok 2: nvidia-container-toolkit ---
if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
    echo "[2/3] Instalacja nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "    nvidia-container-toolkit zainstalowany."
else
    echo "[2/3] nvidia-container-toolkit już zainstalowany - pomijam."
fi

# --- Krok 3: Budowa obrazu Docker ---
echo "[3/3] Budowa obrazu Docker..."
echo "    UWAGA: Pierwsze uruchomienie pobiera Isaac Sim (~25GB)."
echo "    Czas: 30-90 minut w zależności od łącza."
echo ""
cd "$REPO_DIR"
docker compose -f docker/docker-compose.yml build

echo ""
echo "========================================"
echo "  Instalacja zakończona!"
echo "========================================"
echo ""
echo "Uruchomienie środowiska:"
echo "  cd $REPO_DIR && make enter"
echo ""
echo "Jeśli był to pierwszy raz z Dockerem - wyloguj się i zaloguj ponownie."
