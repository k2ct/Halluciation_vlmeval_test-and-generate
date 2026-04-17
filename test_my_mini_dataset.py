import pandas as pd
import base64

def encode_image_file_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

samples = [
    {
        "index": 1,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_1.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "person, scrubs, block, site, scaffolding, baby"
    },
    {
        "index": 2,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_2.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "person, helmet, road, truck"
    },
    {
        "index": 3,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_3.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "woman, desk, laptop, chair"
    }
]

df = pd.DataFrame(samples)
df.to_csv("/root/autodl-tmp/LMUData/my_mini_dataset.tsv", sep="\t", index=False)
print("done")