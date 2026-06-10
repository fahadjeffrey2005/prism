from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json
from nuscenes.nuscenes import NuScenes

COT_PROMPT = """You are an autonomous vehicle perception system. Analyze this driving scene carefully using the following steps:

STEP 1 - SCAN: Describe what you see from left to right across the entire image. Include background, foreground, near and far objects.

STEP 2 - IDENTIFY ACTORS: List every person, vehicle, cyclist you can see. For each one state: type, approximate distance (near/mid/far), and what they appear to be doing.

STEP 3 - ASSESS INTENT: For each moving or potentially moving actor, what are they likely to do in the next 3 seconds?

STEP 4 - IDENTIFY RISKS: What are the top 3 risks in this scene right now?

STEP 5 - DECIDE: Based on your analysis, what should the autonomous vehicle do?
Choose from: CLEAR (full speed) / MONITOR (full speed, watch) / EASE (gentle decel) / SLOW (meaningful decel) / CAUTION (prepare to stop) / YIELD (near stop) / STOP (full stop)

STEP 6 - OUTPUT: Summarize in JSON:
{"total_pedestrians": N, "total_vehicles": N, "decision": "X", "primary_risk": "Y", "reasoning": "Z"}"""


def run_eval(model, processor, nusc, scene_idx):
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
                {"type": "text", "text": COT_PROMPT}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=500, do_sample=False)

    response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("=" * 70)
    print("Scene " + str(scene_idx) + ": " + scene["description"])
    print("Ground Truth: " + str(ped_count) + " peds  " + str(veh_count) + " vehicles")
    print("")
    print("VLM Chain of Thought:")
    print(response)

    try:
        start = response.rfind("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            print("")
            print("PARSED OUTPUT:")
            print("  Pedestrians detected: " + str(parsed.get("total_pedestrians", "?")))
            print("  Vehicles detected:    " + str(parsed.get("total_vehicles", "?")))
            print("  Decision:             " + str(parsed.get("decision", "?")))
            print("  Primary risk:         " + str(parsed.get("primary_risk", "?")))
            ped_error = abs(parsed.get("total_pedestrians", 0) - ped_count)
            print("  Ped count error:      " + str(ped_error) + " (GT=" + str(ped_count) + ")")
    except Exception as e:
        print("JSON parse failed: " + str(e))


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

for scene_idx in [2, 3, 4, 5]:
    run_eval(model, processor, nusc, scene_idx)
