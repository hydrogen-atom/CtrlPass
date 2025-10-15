import json

# 读取原始JSON文件
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为ChatML格式的JSONL
with open('outputml.jsonl', 'w', encoding='utf-8') as f_out:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # 构造单轮对话
            messages = [
                {"role": "user", "content": pattern},
                {"role": "assistant", "content": intent["responses"][0]}  # 取第一个回答
            ]
            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')