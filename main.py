import math

# Menghitung entropy dataset
def entropy(data):
    total = len(data)
    label_counts = {}
    for row in data:
        label = row[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    ent = 0.0
    for label in label_counts:
        prob = label_counts[label] / total
        ent -= prob * math.log2(prob)
    return ent

# Memisahkan data berdasarkan nilai fitur tertentu
def split_data(data, feature_index, value):
    subset = []
    for row in data:
        if row[feature_index] == value:
            reduced_row = row[:feature_index] + row[feature_index + 1:]
            subset.append(reduced_row)
    return subset

# Menentukan fitur terbaik dengan information gain
def best_split(data):
    base_entropy = entropy(data)
    best_gain = 0
    best_feature = -1
    num_features = len(data[0]) - 1
    for i in range(num_features):
        values = set(row[i] for row in data)
        new_entropy = 0.0
        for val in values:
            subset = split_data(data, i, val)
            prob = len(subset) / len(data)
            new_entropy += prob * entropy(subset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature

# Membangun tree rekursif
def build_tree(data, labels):
    class_list = [row[-1] for row in data]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return max(set(class_list), key=class_list.count)

    best_feat = best_split(data)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}
    feat_values = set(row[best_feat] for row in data)
    for value in feat_values:
        sub_labels = labels[:best_feat] + labels[best_feat + 1:]
        sub_data = split_data(data, best_feat, value)
        subtree = build_tree(sub_data, sub_labels)
        tree[best_feat_label][value] = subtree
    return tree

# Fungsi mencetak tree dengan indentasi
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    for key, branches in tree.items():
        print(indent + str(key))
        for value, subtree in branches.items():
            print(indent + "├─" + str(value) + ":")
            print_tree(subtree, indent + "│  ")

# Contoh data dan label
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No'],
    ['Sunny', 'Mild', 'High', 'Strong', 'No']
]
labels = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Membangun pohon dan mencetak
tree = build_tree(data, labels)
print("Decision Tree:")
print_tree(tree)
