from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import argparse
import json
from tqdm import tqdm

def load_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    model.cuda()

    return tokenizer, model

def predict(model, tokenizer, input_text):

    text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

    with torch.no_grad():
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False  # Greedy search
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def main():
    parser = argparse.ArgumentParser(description="Command line interface for dialogue understanding.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the pretrained model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    lora_path = args.lora_path

    tokenizer, model = load_model(model_path, lora_path)
    test_data = json.load(open(data_path + "/test.json"))

    all_outputs = []
    for data in tqdm(test_data):
        response = predict(model, tokenizer, data['text'])
        all_outputs.append(response)
    
    with open(args.output_path, "w") as f:
        json.dump(all_outputs, f, indent=4)

if __name__ == "__main__":
    main()
