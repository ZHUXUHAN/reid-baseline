with open("/home/zxh/train_list.txt", "r+") as f:
    lines = f.readlines()
label_image = {}
for line in lines:
    name, label = line.split(" ")
    label = label[:-1]
    name = name.split('/')[1].split('.')[0]
    # print(name," ",label)
    if label in label_image.keys():
        label_image[label].append(name)
    else:
        label_image[label] = [name]

# print(label_image.keys())
print(len(label_image.keys()))

numbers = dict()

for k, v in label_image.items():
    # print(f"{k} {len(v)}")
    if len(v) in numbers.keys():
        numbers[len(v)].append(k)
    else:
        numbers[len(v)] = [k]

sorted_keys = sorted(numbers)

less_4 = []
less_20_more_4 = []
for key in sorted_keys:
    if key <= 20 and key >= 4:
        less_20_more_4 += numbers[key]
    elif key < 4:
        less_4 += numbers[key]
print(len(set(less_20_more_4)))
print(len(set(less_4)))
with open("less_4.txt", 'w') as f:
    f.write(str(less_4))
with open("less_20_more_4.txt", 'w') as f:
    f.write(str(less_20_more_4))

