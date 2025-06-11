import math

# Fungsi untuk menghitung entropy dari data
def entropy(data):
    total = len(data)  # Menghitung total jumlah data
    label_counts = {}  # Menyimpan jumlah setiap label
    for row in data:
        label = row[-1]  # Label adalah kolom terakhir (Accepted)
        label_counts[label] = label_counts.get(label, 0) + 1  # Hitung frekuensi setiap label
    
    ent = 0.0  # Variabel untuk menyimpan nilai entropy
    for label in label_counts:
        prob = label_counts[label] / total  # Probabilitas kemunculan label
        if prob != 0:  # Menghindari perhitungan log(0) yang tidak terdefinisi
            ent -= prob * math.log2(prob)  # Rumus entropy Shannon
    return round(ent, 4)

# Fungsi untuk membagi data berdasarkan nilai fitur tertentu
def split_data(data, feature_index, value):
    subset = []  # Menyimpan subset data berdasarkan nilai fitur
    for row in data:
        if row[feature_index] == value:  # Memilih data yang memiliki nilai fitur yang sama
            reduced_row = row[:feature_index] + row[feature_index + 1:]  # Menghapus fitur yang sedang dipertimbangkan
            subset.append(reduced_row)
    return subset

# Fungsi untuk menghitung Information Gain dari suatu fitur
def information_gain(data, feature_index, base_entropy):
    feature_values = set(row[feature_index] for row in data)  # Nilai unik dari fitur yang dipilih
    new_entropy = 0.0  # Variabel untuk menyimpan entropy baru setelah membagi data
    
    # Menghitung entropy setelah membagi data berdasarkan nilai fitur
    for value in feature_values:
        subset = split_data(data, feature_index, value)  # Data subset berdasarkan nilai fitur
        prob = len(subset) / len(data)  # Probabilitas subset
        new_entropy += prob * entropy(subset)  # Menambahkan entropy subset ke dalam total entropy baru
    
    # Menghitung Information Gain
    return round(base_entropy - new_entropy, 4)

# Fungsi untuk menampilkan perhitungan entropy dari sebuah fitur
def calculate_feature_entropy(data, feature_values, feature_name, feature_index):
    print(f"{feature_name}:")
    feature_entropy = {}
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]  # Use correct feature index
        feature_entropy[value] = entropy(subset)
        yes_value = sum([1 for row in subset if row[-1] == 'Yes'])
        no_value = len(subset) - yes_value
        print(f"Kolom {value}:")
        print(f"Yes: {yes_value}/{len(subset)}")
        print(f"No: {no_value}/{len(subset)}")
        print(f"Entropy {value}: {feature_entropy[value]}")
    return feature_entropy

# Fungsi untuk menghitung weighted entropy dan information gain berdasarkan perhitungan entropy fitur
def calculate_weighted_entropy(data, feature_values, feature_entropy, feature_index):
    weighted_entropy = sum([
        (len([row for row in data if row[feature_index] == value]) / len(data)) * feature_entropy[value]
        for value in feature_values
    ])
    return round(weighted_entropy, 4)


# Fungsi untuk menghitung informasi gain berdasarkan entropy per fitur
def calculate_information_gain(base_entropy, weighted_entropy):
    return round(base_entropy - weighted_entropy, 4)

# Fungsi untuk membangun decision tree menggunakan ID3
def id3(data, features):
    labels = [row[-1] for row in data]
    if len(set(labels)) == 1:
        return labels[0]

    if not features:
        return max(set(labels), key=labels.count)

    gains = [information_gain(data, i, entropy(data)) for i in range(len(features))]
    best_feature_index = gains.index(max(gains))

    best_feature = features[best_feature_index]
    tree = {best_feature: {}}
    feature_values = set(row[best_feature_index] for row in data)
    sub_features = features[:best_feature_index] + features[best_feature_index + 1:]

    for value in feature_values:
        subset = [row for row in data if row[best_feature_index] == value]
        tree[best_feature][value] = id3(subset, sub_features)

    return tree

# Fungsi untuk mencetak pohon keputusan dalam format yang mudah dibaca
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    for key, branches in tree.items():
        print(indent + str(key))
        for value, subtree in branches.items():
            print(indent + "├─" + str(value) + ":")
            print_tree(subtree, indent + "│  ")

# Main function
def main(data, features):
    # Hitung Entropy untuk kelas "Accepted"
    base_entropy = entropy(data)
    yes_count = sum([1 for row in data if row[-1] == 'Yes'])
    no_count = len(data) - yes_count

    # Perhitungan Total Entropy
    print(f"Perhitungan Total Entropy:")
    print(f"Yes: {yes_count}/{len(data)}")
    print(f"No: {no_count}/{len(data)}")
    print(f"Entropy Class: {base_entropy}\n")

    # kalkulasi Information Gain untuk tiap fitur dan entropy nya
    gpa_values = ['Good', 'Average', 'Poor']
    gpa_entropy = calculate_feature_entropy(data, gpa_values, "GPA", 0)
    weighted_entropy_gpa = calculate_weighted_entropy(data, gpa_values, gpa_entropy, 0)
    information_gain_gpa = calculate_information_gain(base_entropy, weighted_entropy_gpa)

    print(f"Weighted Entropy GPA: {weighted_entropy_gpa}")
    print(f"Information Gain GPA: {information_gain_gpa}\n")

    psychology_values = ['Strong', 'Moderate', 'Weak']
    psychology_entropy = calculate_feature_entropy(data, psychology_values, "Psychology", 1)
    weighted_entropy_psy = calculate_weighted_entropy(data, psychology_values, psychology_entropy, 1)
    information_gain_psy = calculate_information_gain(base_entropy, weighted_entropy_psy)

    print(f"Weighted Entropy Psychology: {weighted_entropy_psy}")
    print(f"Information Gain Psychology: {information_gain_psy}\n")

    interview_values = ['Proper', 'Unsuitable']
    interview_entropy = calculate_feature_entropy(data, interview_values, "Interview", 2)
    weighted_entropy_int = calculate_weighted_entropy(data, interview_values, interview_entropy, 2)
    information_gain_int = calculate_information_gain(base_entropy, weighted_entropy_int)

    print(f"Weighted Entropy Interview: {weighted_entropy_int}")
    print(f"Information Gain Interview: {information_gain_int}\n")

    # Membangun pohon keputusan menggunakan ID3
    tree = id3(data, features)
    print("Pohon Keputusan:")
    print_tree(tree)

# Data sampel (Fitur: GPA, Psychology, Interview, Target: Accepted)
data = [
    ['Good', 'Strong', 'Proper', 'Yes'],
    ['Good', 'Moderate', 'Proper', 'Yes'],
    ['Good', 'Moderate', 'Unsuitable', 'Yes'],
    ['Good', 'Weak', 'Unsuitable', 'No'],
    ['Average', 'Strong', 'Proper', 'Yes'],
    ['Average', 'Moderate', 'Proper', 'Yes'],
    ['Average', 'Moderate', 'Unsuitable', 'Yes'],
    ['Average', 'Weak', 'Unsuitable', 'No'],
    ['Poor', 'Strong', 'Proper', 'Yes'],
    ['Poor', 'Moderate', 'Unsuitable', 'No'],
    ['Poor', 'Weak', 'Proper', 'Yes'],
]

# Fitur
features = ['GPA', 'Psychology', 'Interview']

# Menjalankan program
main(data, features)
