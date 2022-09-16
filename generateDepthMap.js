import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-core";
import { node } from "@tensorflow/tfjs-node";
import images from "images";
import * as fs from "fs";
import path from "path";

const main = async () => {
  if (process.argv.length != 4) {
    console.error(
      "[!] The number of command line arguments is incompatible with the script. \nTwo Arguments are supported. \nPrototype: npm run inference <SRC_IMG_PATH> <DES_PATH>"
    );
    return;
  }
  const [src_img, dest_path] = process.argv.slice(2);
  console.log(
    `Source Image Path: ${src_img} \nDestination file path: ${dest_path}`
  );

  const image = images.loadFromFile(src_img);
  const width = image.width();
  const height = image.height();

  const loadedBuffer = fs.readFileSync(
    path.join(dest_path, "depth_map_tensor.bin")
  );
  const newFloat32Array = new Float32Array(loadedBuffer.buffer);

  const tensor = tf.tensor(newFloat32Array, [height, width], "float32");

  node.encodeJpeg(tensor.mul(255).expandDims(2), "grayscale", 100).then((f) => {
    fs.writeFileSync(path.join(dest_path, "depthMap.jpeg"), f);
    console.log("Basic JPEG 'depthMap.jpeg' written");
  });
};

main();

// Tensor {
//     kept: false,
//     isDisposedInternal: false,
//     shape: [ 2048, 1536 ],
//     dtype: 'float32',
//     size: 3145728,
//     strides: [ 1536 ],
//     dataId: { id: 526 },
//     id: 458,
//     rankType: '2',
//     scopeId: 399
//   }
