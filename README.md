ONNX Runtime for Web on-edge training examples.

### Usage

First, we create the onnx model files:
```bash
python -m pip install torch torchvision onnx onnxruntime-training
python create_onnx_files.py
```

Then, run the frontend:
```bash
npm install
npm run dev
```
