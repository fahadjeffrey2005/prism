from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
from nuscenes.nuscenes import NuScenes

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True
).cuda()
model.eval()

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="/home/koushik-test/prism_data/datasets/nuscenes",
    verbose=False
)

for scene_idx in [6, 7, 8, 9]:
    scene = nusc.scene[scene_idx]
    sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    img_path = "/home/koushik-test/prism_data/datasets/nuscenes/" + sd["filename"]
    image = Image.open(img_path).convert("RGB")

    ped_count = sum(
        1 for a in sample["anns"]
        if "pedestrian" in nusc.get("sample_annotation", a)["category_name"]
    )
    veh_count = sum(
        1 for a in sample["anns"]
        if "vehicle" in nusc.get("sample_annotation", a)["category_name"]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Count pedestrians and vehicles visible. Reply only in JSON format: {\"pedestrians\": N, \"vehicles\": N, \"action\": \"CLEAR/SLOW/STOP\", \"reason\": \"brief reason\"}"}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("Scene " + str(scene_idx) + ": " + scene["description"][:50])
    print("GT: " + str(ped_count) + " peds  " + str(veh_count) + " vehicles")
    print("VLM: " + response)
    print("------------------------------------------------------------")
