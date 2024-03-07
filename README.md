Dependencies:

```bash
python -m pip install torch torchvision onnx onnxruntime-training
npm install
```

ORT Training for Web works on a simple 2-layer MLP:

```bash
python create_onnx_files.py "mlp"
npm run dev
```

But not for ResNet:

```bash
rm -r models/
python create_onnx_files.py "resnet"
npm run dev
```
