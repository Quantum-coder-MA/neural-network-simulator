"# Model Architecture Specification" 
- Input: 3x64x64
- ConvBlock1: Conv2d(3→16, 3×3), ReLU, MaxPool
- ConvBlock2: Conv2d(16→32, 3×3), ReLU, MaxPool
- ConvBlock3: Conv2d(32→64, 3×3), ReLU, GlobalAvgPool
- FC1: 64 → 32 → ReLU
- FC2: 32 → 2
- Loss: CrossEntropy
