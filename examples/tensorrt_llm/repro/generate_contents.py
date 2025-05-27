import json

contents = []
with open("/Users/rihuo/Downloads/profile_export_1.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    payloads = [item["request_inputs"]["payload"] for item in data["experiments"][0]["requests"]]

    for payload in payloads:
        request = json.loads(payload)
        contents.append({"content": request["messages"][0]["content"]})


with open("output.json", "w") as f:
    json.dump(contents, f, indent=2)
