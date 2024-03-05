import * as ort from "onnxruntime-web/training";

function imgToImageData(img) {
    const canvas = new OffscreenCanvas(img.width, img.height);
    const context = canvas.getContext("2d");
    context.drawImage(img, 0, 0);
    return context.getImageData(0, 0, img.width, img.height);
}

async function main() {
    const session = await ort.TrainingSession.create({
        checkpointState: "checkpoint",
        trainModel: "training_model.onnx",
        evalModel: "eval_model.onnx",
        optimizerModel: "optimizer_model.onnx",
    });

    const input = document.getElementById("input");
    const preview = document.getElementById("preview");

    input.onchange = (event) => {
        preview.src = URL.createObjectURL(event.target.files[0]);

        preview.width = 224;
        preview.height = 224;
        preview.style = "visibility: visible;";

        preview.onload = () => URL.revokeObjectURL(preview.src); // free memory

        console.log(imgToImageData(preview));
    };
}

await main();
