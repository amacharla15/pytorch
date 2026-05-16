import argparse
import os
import json

import torch
import torch.nn as nn
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/compile_graph_capture")
    parser.add_argument("--text", type=str, default="This movie was excellent and emotional.")
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def write(path, content):
    with open(path, "w") as f:
        f.write(str(content))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    base_model.eval()

    model = LogitsWrapper(base_model)
    model.eval()

    encoded = tokenizer(
        args.text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    batch_info = {
        "input_text": args.text,
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "model_dir": args.model_dir
    }

    with open(os.path.join(args.output_dir, "batch_info.json"), "w") as f:
        json.dump(batch_info, f, indent=2)

    try:
        explanation = torch._dynamo.explain(model)(input_ids, attention_mask)
        write(os.path.join(args.output_dir, "dynamo_explain.txt"), explanation)
    except Exception as e:
        write(os.path.join(args.output_dir, "dynamo_explain_error.txt"), repr(e))

    counter = {"n": 0}

    def capture_backend(gm, example_inputs):
        i = counter["n"]
        counter["n"] += 1

        write(os.path.join(args.output_dir, f"graph_{i}.txt"), gm.graph)
        write(os.path.join(args.output_dir, f"graph_{i}_code.py"), gm.code)

        try:
            dot_path = os.path.join(args.output_dir, f"graph_{i}.dot")
            with open(dot_path, "w") as f:
                f.write("digraph FXGraph {\n")
                for node in gm.graph.nodes:
                    label = str(node).replace('"', "'")
                    f.write(f'  "{node.name}" [label="{label}\\n{node.op}\\n{node.target}"];\n')
                    for arg in node.all_input_nodes:
                        f.write(f'  "{arg.name}" -> "{node.name}";\n')
                f.write("}\n")
        except Exception as e:
            write(os.path.join(args.output_dir, f"graph_{i}_dot_error.txt"), repr(e))

        return gm.forward

    try:
        torch._dynamo.reset()
        compiled_model = torch.compile(model, backend=capture_backend, fullgraph=False)

        with torch.no_grad():
            logits = compiled_model(input_ids, attention_mask)

        result = {
            "captured_graph_count": counter["n"],
            "logits": logits.detach().cpu().tolist(),
            "prediction": int(torch.argmax(logits, dim=1).item()),
            "note": "These are TorchDynamo FX graphs captured with a custom backend. This captures the graph region sent to the backend, not a final Inductor kernel visualization."
        }

        with open(os.path.join(args.output_dir, "capture_summary.json"), "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        write(os.path.join(args.output_dir, "capture_error.txt"), repr(e))

    try:
        exported = torch.export.export(model, (input_ids, attention_mask), strict=False)
        write(os.path.join(args.output_dir, "torch_export_graph.txt"), exported.graph_module.graph)
        write(os.path.join(args.output_dir, "torch_export_code.py"), exported.graph_module.code)
    except Exception as e:
        write(os.path.join(args.output_dir, "torch_export_error.txt"), repr(e))

    print("Saved graph capture files to:", args.output_dir)


if __name__ == "__main__":
    main()
