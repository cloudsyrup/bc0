$ErrorActionPreference = 'Stop'

# Run from the project root after installing requirements.txt.
python -m PyInstaller --noconfirm --clean --onefile --name bc0 `
  --collect-all chess --collect-all numpy engine.py

Write-Host "Built dist\bc0.exe"
