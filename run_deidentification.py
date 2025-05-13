import argparse
import os
import glob
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
import torchvision.transforms as transforms

def main(prompt, neg_prompt, strength, image_folder, depth_folder, result_destination):
    os.makedirs(result_destination, exist_ok=True)

    # Load the pipeline
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
    ).to("cuda")

    transform = transforms.ToTensor()

    # Get image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_folder, "*.png")))

    image_index_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    depth_index_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in depth_paths}

    print(f"Total images found: {len(image_paths)}")
    print(f"Total depth maps found: {len(depth_paths)}")

    batch_size = 100
    counter = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        print("Batch counter:", counter)
        counter += 1

        batch_keys = list(image_index_dict.keys())[i:i + batch_size]
        images1 = [Image.open(image_index_dict[k]) for k in batch_keys if k in depth_index_dict]
        images2depth = [Image.open(depth_index_dict[k]).convert("L") for k in batch_keys if k in depth_index_dict]

        tensor_images1 = [transform(img).unsqueeze(0) for img in images1]
        tensor_images2depth = [transform(img) for img in images2depth]

        for idx, key in enumerate(batch_keys):
            if key not in depth_index_dict:
                continue

            image_name = f"{key}.png"
            save_to = os.path.join(result_destination, image_name)

            if not os.path.exists(save_to):
                tensor_img = tensor_images1[idx]
                tensor_depth = tensor_images2depth[idx]

                model_result = pipe(
                    prompt=prompt,
                    image=tensor_img,
                    negative_prompt=neg_prompt,
                    strength=strength,
                    depth_map=tensor_depth,
                )
                model_result.images[0].save(save_to)
            else:
                print(f"Skipping {image_name}, already exists.")

    print(f"Generated images saved to {result_destination}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion Depth2Img pipeline")
    parser.add_argument('--prompt', type=str, default="picture of a human", help="Text prompt to guide image generation")
    parser.add_argument('--negprompt', type=str, default="bad anatomy, extra eyes", help="Negative prompt to avoid undesired results")
    parser.add_argument('--strength', type=float, default=0.87, help="Denoising strength (0.0 to 1.0)")
    parser.add_argument('--image_folder', type=str, default="./temp/k-anonymity", help="Path to the folder containing input images")
    parser.add_argument('--depth_folder', type=str, default="./temp/depth", help="Path to the folder containing depth maps")
    parser.add_argument('--result_destination', type=str, default="./results/generated/", help="Path to save generated images")

    args = parser.parse_args()

    main(
        prompt=args.prompt,
        neg_prompt=args.negprompt,
        strength=args.strength,
        image_folder=args.image_folder,
        depth_folder=args.depth_folder,
        result_destination=args.result_destination
    )
