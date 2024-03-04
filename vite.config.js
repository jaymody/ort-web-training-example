import { viteStaticCopy } from "vite-plugin-static-copy";

export default {
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/onnxruntime-web/dist/*.wasm",
                    dest: "",
                },
            ],
        }),
    ],
};
