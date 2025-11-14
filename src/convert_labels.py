import ast

def load_and_convert_labels(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        content = f.read().strip()
        # Parse cả file thành dict
        imagenet_classes_dict = ast.literal_eval(content)

    # Chuyển dict sang list theo thứ tự ID
    imagenet_classes = [imagenet_classes_dict[i] for i in range(len(imagenet_classes_dict))]

    # Xuất ra file .txt mới
    with open(output_path, "w", encoding="utf-8") as f:
        for name in imagenet_classes:
            f.write(name + "\n")

    print(f"Đã xuất {len(imagenet_classes)} nhãn sang {output_path}")


if __name__ == "__main__":
    load_and_convert_labels("imagenet1000_clsidx_to_labels.txt", "imagenet_labels_clean.txt")
