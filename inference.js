import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";
import "@tensorflow-models/body-segmentation";
import * as depthEstimation from "@tensorflow-models/depth-estimation";
import * as fs from "fs";
import images from "images";
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

  console.log("[1] Loading depth estimation model ....");
  const model = depthEstimation.SupportedModels.ARPortraitDepth;
  const estimator = await depthEstimation.createEstimator(model);
  const estimationConfig = {
    minDepth: 0, // The minimum depth value outputted by the estimator.
    maxDepth: 1, // The maximum depth value outputted by the estimator.
  };

  console.log("[2] Loading image .....");
  const image = images.loadFromFile(src_img);
  const width = image.width();
  const height = image.height();
  const raw = image.toBuffer(images.TYPE_RAW);
  const pixels = new Uint8Array(raw.buffer, 12);
  const out = await tf.browser.fromPixels({ data: pixels, width, height });

  console.log("[3] Performing inference ....");
  const depthMap = await estimator.estimateDepth(out, estimationConfig);

  console.log("Converting to saveable formats ....");

  const dpm_tensor = await depthMap.toTensor();
  const dump_arr = dpm_tensor.dataSync();

  try {
    fs.writeFileSync(
      path.join(dest_path, "depth_map_tensor.bin"),
      Buffer.from(dump_arr.buffer)
    );
    console.log("[4] Depth Map Tensor file written successfully\n");
  } catch (err) {
    console.log(
      `An error occurred while saving the tensor file. The error is as follows. \n${err}`
    );
  }

  console.log("[5] Script completed. Exiting.");
};

main();
