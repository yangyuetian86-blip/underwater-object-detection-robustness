import json

# 请替换成你两个 JSON 的真实路径
file1 = r'runs\detect\val-16\predictions.json'
file2 = r'runs\detect\val-17\predictions.json'

with open(file1, 'r') as f1, open(file2, 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# 取第一个目标的得分对比
score1 = data1[0]['score']
score2 = data2[0]['score']

print(f"--- 微观置信度对比 ---")
print(f"Fold A 第一个目标得分: {score1}")
print(f"Fold B 第一个目标得分: {score2}")
print(f"绝对差值: {abs(score1 - score2)}")