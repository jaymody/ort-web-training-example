import * as ort from "onnxruntime-web/training";

function indexOfMax(arr) {
    var max = arr[0];
    var maxIdx = 0;
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIdx = i;
            max = arr[i];
        }
    }
    return maxIdx;
}

function argmax(t) {
    const preds = [];
    const [batchSize, numClasses] = t.dims;
    for (let i = 0; i < batchSize; i++) {
        const logits = t.data.slice(i * numClasses, (i + 1) * numClasses);
        const pred = indexOfMax(logits);
        preds.push(pred);
    }
    return preds;
}

function imageDataToFlatGrayscaleFloatArr(imageData) {
    const arr = new Float32Array(imageData.width * imageData.height);

    const data = imageData.data;
    for (var i = 0; i < data.length / 4; i++) {
        const j = i * 4;
        arr[i] = data[j + 0] / 3 + data[j + 1] / 3 + data[j + 2] / 3;
        arr[i] /= 255.0;
    }

    return arr;
}

async function main() {
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;

    const clearBtn = document.createElement("button");
    clearBtn.innerText = "clear canvas";

    const outtext = document.createElement("p");
    outtext.innerText = "loading ...";
    document.body.appendChild(outtext);

    const session = await ort.TrainingSession.create({
        checkpointState: "checkpoint",
        trainModel: "training_model.onnx",
        evalModel: "eval_model.onnx",
        optimizerModel: "optimizer_model.onnx",
    });

    async function getPrediction(imageData) {
        const len = 1;
        const size = 784;
        const input = imageDataToFlatGrayscaleFloatArr(imageData);
        const label = new BigInt64Array([BigInt(7)]);

        const results = await session.runEvalStep({
            input: new ort.Tensor("float32", input, [len, size]),
            labels: new ort.Tensor("int64", label, [len]),
        });

        const preds = argmax(results["output"]);

        return preds[0];
    }

    function clearCanvas() {
        outtext.innerText = "draw a number";
        context.fillStyle = "black";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    const context = canvas.getContext("2d");
    document.addEventListener("mousemove", (event) => {
        // mouse left button must be pressed
        if (event.buttons !== 1) return;

        // xy coordinate in canvas
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // skip if out of bounds
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) return;

        // draw black circle at mouse position
        context.beginPath();
        context.arc(x, y, 1, 0, 2 * Math.PI, false);
        context.lineWidth = 0;
        context.fillStyle = "white";
        context.fill();
    });

    document.addEventListener("mouseup", async (event) => {
        // skip if out of bounds
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) return;

        let imageData = context.getImageData(0, 0, canvas.width, canvas.height);

        const pred = await getPrediction(imageData);

        outtext.innerText = `prediction = ${pred}`;
    });

    clearBtn.onclick = () => clearCanvas();

    clearCanvas();
    document.body.appendChild(canvas);
    document.body.appendChild(document.createElement("br"));
    document.body.appendChild(clearBtn);
}

await main();
