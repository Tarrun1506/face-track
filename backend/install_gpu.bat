@echo off
echo ============================================================
echo   Installing GPU-accelerated PyTorch (CUDA 12.1)
echo   For: NVIDIA RTX 4050 Laptop GPU
echo ============================================================
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
echo.
echo Installing other dependencies...
pip install flask==2.3.3 flask-cors==4.0.0 numpy==1.24.3 opencv-python==4.8.0.76 facenet-pytorch==2.5.3 scipy==1.10.1 pillow==10.0.0
echo.
echo ============================================================
echo   Done! Run the app with: python app.py
echo ============================================================
pause
