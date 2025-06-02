from conversation import get_conv_template
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--new-sys-message", type=str, default=" ")
parser.add_argument("--output-path", type=str, required=True)
args = parser.parse_args()

with open(args.input_file, "r", encoding="utf-8") as f:
    raw_data = json.loads(f.read())

plain_text = []

sources = [example["conversations"] for example in raw_data]

for i, source in enumerate(sources): # instance loop

    conv = get_conv_template(args.template)
    conv.system_message = args.new_sys_message

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conv.messages = []
    for j, sentence in enumerate(source): # dialog loop
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    
    plain_text.append(conv.get_prompt())

with open(args.output_path, "w+", encoding="utf-8") as f:
    f.write(json.dumps(plain_text))