import io
import os

import onnx
import torch
from onnxruntime.training import artifacts
from torchvision import transforms


class Model(torch.nn.Module):
    def __init__(self, model_repo, model_name):
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.network = torch.hub.load(model_repo, model_name, pretrained=True)

    def forward(self, x):
        x = self.normalize(x)
        x = self.network(x)
        return x


def main(out_dir="models/"):
    os.makedirs(out_dir, exist_ok=True)

    # model stuff
    pt_model = Model("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0")
    example_input = (torch.randn(1, 3, 224, 224),)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # normal onnx model export
    torch.onnx.export(
        pt_model,
        example_input,
        os.path.join(out_dir, "model.onnx"),
        input_names=input_names,
        output_names=output_names,
    )

    # training onnx model export
    f = io.BytesIO()
    torch.onnx.export(
        pt_model,
        example_input,
        f,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.TRAINING,
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=False,
    )
    onnx_model = onnx.load_from_string(f.getvalue())
    requires_grad = [
        name for name, param in pt_model.named_parameters() if param.requires_grad
    ]
    frozen_params = [
        name for name, param in pt_model.named_parameters() if not param.requires_grad
    ]
    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.AdamW,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        additional_output_names=output_names,
        artifact_directory=out_dir,
    )


if __name__ == "__main__":
    main()
