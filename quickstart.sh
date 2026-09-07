#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# quickstart.sh — One-shot setup and launch for CropScan AI
# Usage: bash quickstart.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on first error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'   # No Colour

echo -e "\n${GREEN}🌿  CropScan AI — Quick Start${NC}\n"

# 1. Virtual environment
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}▶ Creating virtual environment …${NC}"
  python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 2. Install deps
echo -e "${YELLOW}▶ Installing dependencies …${NC}"
pip install -q -r requirements.txt

# 3. Generate dataset (if not already there)
if [ ! -d "dataset/train" ]; then
  echo -e "${YELLOW}▶ Generating synthetic dataset …${NC}"
  python generate_dataset.py
fi

# 4. Train model (if not already there)
if [ ! -f "model/crop_disease_model.h5" ]; then
  echo -e "${YELLOW}▶ Training CNN model (this may take a few minutes) …${NC}"
  python train.py
fi

# 5. Launch server
echo -e "\n${GREEN}✅  Setup complete! Starting web server …${NC}"
echo -e "${GREEN}   Open → http://localhost:5000${NC}\n"
python app.py
