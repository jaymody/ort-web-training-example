import * as ort from "onnxruntime-web/training";

import { imgToImageData, argmax } from "./utils.js";

async function main() {
    const session = await ort.TrainingSession.create({
        checkpointState: "checkpoint",
        trainModel: "training_model.onnx",
        evalModel: "eval_model.onnx",
        optimizerModel: "optimizer_model.onnx",
    });

    async function getPrediction(imageData) {
        const data = imageData.data;

        const len = 1;
        const size = 784;
        const input = new Float32Array(size);
        const label = new BigInt64Array([BigInt(7)]);

        for (var i = 0; i < data.length / 4; i++) {
            const j = i * 4;
            input[i] = (data[j + 0] + data[j + 1] + data[j + 3]) / 3;
            input[i] /= 255.0;
        }

        const results = await session.runEvalStep({
            input: new ort.Tensor("float32", input, [len, size]),
            labels: new ort.Tensor("int64", label, [len]),
        });

        const preds = argmax(results["output"]);

        return preds[0];
    }

    const input = document.getElementById("input");
    const preview = document.getElementById("preview");
    const outtext = document.getElementById("outtext");

    input.onchange = async (event) => {
        preview.src = URL.createObjectURL(event.target.files[0]);

        preview.width = 28;
        preview.height = 28;
        preview.style = "visibility: visible;";

        preview.onload = () => URL.revokeObjectURL(preview.src); // free memory

        const imageData = imgToImageData(preview);

        const pred = await getPrediction(imageData);

        outtext.innerText = `prediction = ${pred}`;
    };
}

await main();
