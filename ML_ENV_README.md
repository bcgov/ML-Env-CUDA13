# ML-Env-CUDA13 (GPU Ready)

**GPU**: RTX 2000 Ada  
**CUDA**: 13.0.2 (System)  
**PyTorch**: 2.9.0+cu130 (CUDA 13.0)  
**TensorFlow**: 2.17.0 (CUDA 11.8 via wheels)

## Activate
```powershell
.\cuda_clean_env\Scripts\Activate
```

## Test
```powershell
python test_pytorch.py
python test_tensorflow.py
```

## Reinstall
```powershell
.\setup_ml_env_full.ps1
```

## Deactivate
```powershell
deactivate
```
