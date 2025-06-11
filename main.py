import math

#? Fungsi untuk memprediksi label berdasarkan decision tree dan sampel data
def predict(tree, sample, features):
    if not isinstance(tree, dict):
        return tree  #? Jika node adalah daun (label hasil), kembalikan label
    root = next(iter(tree))  #? Ambil fitur root dari tree
    root_index = features.index(root)  #? Cari indeks fitur root
    value = sample[root_index]  #? Ambil nilai fitur dari sampel
    subtree = tree[root].get(value)  #? Ambil subtree berdasarkan nilai fitur
    if subtree is None:
        return None  #? Tangani jika branch tidak dikenali
    #? Buat sampel dan fitur baru tanpa fitur yang sudah digunakan
    sub_sample = sample[:root_index] + sample[root_index+1:]
    sub_features = features[:root_index] + features[root_index+1:]
    return predict(subtree, sub_sample, sub_features)

#? Fungsi untuk menghitung entropy dari dataset
def entropy(data):
    total = len(data)
    label_counts = {}
    #? Hitung jumlah tiap label
    for row in data:
        label = row[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    ent = 0.0
    #? Hitung entropy menggunakan formula
    for label in label_counts:
        prob = label_counts[label] / total
        if prob != 0:
            ent -= prob * math.log2(prob)
    return round(ent, 4)

#? Fungsi untuk memisahkan data berdasarkan nilai fitur
def split_data(data, feature_index, value):
    subset = []
    for row in data:
        if row[feature_index] == value:
            #? Buang kolom fitur yang digunakan
            reduced_row = row[:feature_index] + row[feature_index + 1:]
            subset.append(reduced_row)
    return subset

#? Fungsi untuk menghitung Information Gain dari sebuah fitur
def information_gain(data, feature_index, base_entropy):
    feature_values = set(row[feature_index] for row in data)
    new_entropy = 0.0
    for value in feature_values:
        subset = split_data(data, feature_index, value)
        prob = len(subset) / len(data)
        new_entropy += prob * entropy(subset)
    return round(base_entropy - new_entropy, 4)

#? Fungsi untuk menampilkan dan menghitung entropy per nilai fitur
def calculate_feature_entropy(data, feature_values, feature_name, feature_index):
    print(f"{feature_name}:")
    feature_entropy = {}
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        feature_entropy[value] = entropy(subset)
        #? Hitung jumlah 'Yes' dan 'No'
        yes_value = sum([1 for row in subset if row[-1] == 'Yes'])
        no_value = len(subset) - yes_value
        #? Tampilkan informasi
        print(f"Kolom {value}:")
        print(f"Yes: {yes_value}/{len(subset)}")
        print(f"No: {no_value}/{len(subset)}")
        print(f"Entropy {value}: {feature_entropy[value]}")
    return feature_entropy

#? Fungsi untuk menghitung weighted entropy berdasarkan proporsi tiap nilai fitur
def calculate_weighted_entropy(data, feature_values, feature_entropy, feature_index):
    weighted_entropy = sum([
        (len([row for row in data if row[feature_index] == value]) / len(data)) * feature_entropy[value]
        for value in feature_values
    ])
    return round(weighted_entropy, 4)

#? Fungsi untuk menghitung information gain dari base entropy dan weighted entropy
def calculate_information_gain(base_entropy, weighted_entropy):
    return round(base_entropy - weighted_entropy, 4)

#? Fungsi utama ID3 untuk membuat pohon keputusan
def id3(data, features):
    labels = [row[-1] for row in data]
    
    #? Basis: jika semua label sama, kembalikan label tersebut
    if len(set(labels)) == 1:
        return labels[0]
    
    #? Basis: jika tidak ada fitur tersisa, kembalikan label mayoritas
    if not features:
        return max(set(labels), key=labels.count)

    #? Hitung information gain untuk semua fitur
    gains = [information_gain(data, i, entropy(data)) for i in range(len(features))]
    best_feature_index = gains.index(max(gains))
    best_feature = features[best_feature_index]

    #? Buat node root
    tree = {best_feature: {}}
    feature_values = set(row[best_feature_index] for row in data)
    
    #? Fitur baru tanpa fitur terbaik
    sub_features = features[:best_feature_index] + features[best_feature_index + 1:]

    #? Rekursi ke tiap nilai dari fitur terbaik
    for value in feature_values:
        subset = [row for row in data if row[best_feature_index] == value]
        tree[best_feature][value] = id3(subset, sub_features)
    
    return tree

#? Fungsi untuk mencetak struktur pohon keputusan
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    for key, branches in tree.items():
        print(indent + str(key))
        for value, subtree in branches.items():
            print(indent + "├─" + str(value) + ":")
            print_tree(subtree, indent + "│  ")

#? Fungsi evaluasi akurasi pohon terhadap data uji
def evaluate(tree, test_data, features):
    correct = 0
    for row in test_data:
        prediction = predict(tree, row[:-1], features)
        actual = row[-1]
        print(f"Input: {row[:-1]} → Predicted: {prediction}, Actual: {actual}")
        if prediction == actual:
            correct += 1
    accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

#? Fungsi utama untuk menjalankan training dan evaluasi
def main(data, features):
    base_entropy = entropy(data)
    yes_count = sum([1 for row in data if row[-1] == 'Yes'])
    no_count = len(data) - yes_count

    #? Tampilkan informasi entropy awal
    print(f"Perhitungan Total Entropy:")
    print(f"Yes: {yes_count}/{len(data)}")
    print(f"No: {no_count}/{len(data)}")
    print(f"Entropy Class: {base_entropy}\n")

    #? === Hitung dan tampilkan entropy untuk masing-masing fitur ===

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

    #? Bangun pohon keputusan
    tree = id3(data, features)
    print("Pohon Keputusan:")
    print_tree(tree)

    #? Evaluasi model terhadap data training
    correct = 0
    for row in data:
        prediction = predict(tree, row[:-1], features)
        actual = row[-1]
        if prediction == actual:
            correct += 1
    accuracy = correct / len(data) * 100
    print(f"\nTrain Accuracy: {accuracy:.2f}%")

    #? Evaluasi terhadap data uji (tambahan)
    print("\nEvaluasi terhadap data uji:")
    test_data = [
        ['Good', 'Moderate', 'Proper', 'Yes'],
        ['Poor', 'Weak', 'Unsuitable', 'No'],
        ['Average', 'Strong', 'Unsuitable', 'Yes'],
        ['Good', 'Weak', 'Proper', 'Yes'],
        ['Poor', 'Strong', 'Unsuitable', 'No']
    ]
    evaluate(tree, test_data, features)

#? Data training
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

#? Fitur yang digunakan
features = ['GPA', 'Psychology', 'Interview']

#? Jalankan program utama
main(data, features)
