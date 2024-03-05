function imgToImageData(img) {
    const canvas = new OffscreenCanvas(img.width, img.height);
    const context = canvas.getContext("2d");
    context.drawImage(img, 0, 0);
    return context.getImageData(0, 0, img.width, img.height);
}

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

export { imgToImageData, indexOfMax, argmax };
