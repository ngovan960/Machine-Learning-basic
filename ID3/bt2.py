import math

import pandas as pd


TARGET = "Rủi ro tín dụng"
NUMERIC_FEATURES = {"Độ tuổi", "Thu nhập"}


def entropy(labels):
    probabilities = labels.value_counts(normalize=True)
    return -sum(probability * math.log2(probability) for probability in probabilities)


def information_gain(data, feature, threshold=None):
    if threshold is None:
        groups = data.groupby(feature, dropna=False)[TARGET]
    else:
        groups = data.groupby(data[feature] <= threshold, dropna=False)[TARGET]

    weighted_entropy = sum(
        len(group) / len(data) * entropy(group)
        for _, group in groups
    )
    return entropy(data[TARGET]) - weighted_entropy


def best_split(data, features):
    best = None
    for feature in features:
        if feature in NUMERIC_FEATURES:
            values = sorted(data[feature].unique())
            splits = [(left + right) / 2 for left, right in zip(values, values[1:])]
        else:
            splits = [None]

        for threshold in splits:
            gain = information_gain(data, feature, threshold)
            if best is None or gain > best["gain"]:
                best = {"feature": feature, "threshold": threshold, "gain": gain}
    return best


def build_tree(data, features):
    labels = data[TARGET]
    if len(labels.unique()) == 1:
        return {"leaf": labels.iloc[0]}
    if not features:
        return {"leaf": labels.mode()[0]}

    split = best_split(data, features)
    remaining = [feature for feature in features if feature != split["feature"]]
    node = {
        "feature": split["feature"],
        "threshold": split["threshold"],
        "gain": split["gain"],
        "children": {},
    }

    if split["threshold"] is None:
        for value, group in data.groupby(split["feature"], dropna=False):
            node["children"][str(value)] = build_tree(group, remaining)
    else:
        threshold = split["threshold"]
        node["children"][f"<= {threshold:g}"] = build_tree(
            data[data[split["feature"]] <= threshold], remaining
        )
        node["children"][f"> {threshold:g}"] = build_tree(
            data[data[split["feature"]] > threshold], remaining
        )
    return node


def print_tree(node, prefix=""):
    if "leaf" in node:
        print(f"{prefix}=> Rủi ro tín dụng = {node['leaf']}")
        return

    print(f"{prefix}[{node['feature']}, Gain = {node['gain']:.4f}]")
    for condition, child in node["children"].items():
        print(f"{prefix}├── {condition}")
        print_tree(child, prefix + "│   ")


def main():
    data = pd.DataFrame(
        {
            "ID": range(1, 16),
            "Độ tuổi": [25, 40, 35, 27, 31, 36, 48, 26, 33, 29, 38, 44, 42, 28, 30],
            "Hôn nhân": ["Độc thân", "Đã kết hôn", "Từng ly hôn", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Từng ly hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn"],
            "Sở hữu BĐS": ["Ở cùng bố mẹ", "Nhà sở hữu", "Nhà thuê", "Ở cùng bố mẹ", "Nhà thuê", "Nhà sở hữu", "Nhà thuê", "Nhà thuê", "Ở cùng bố mẹ", "Nhà thuê", "Nhà sở hữu", "Nhà sở hữu", "Nhà sở hữu", "Nhà thuê", "Ở cùng bố mẹ"],
            "Thu nhập": [7000000, 18000000, 12000000, 9000000, 6000000, 8000000, 7000000, 8000000, 5000000, 10000000, 15000000, 14000000, 10000000, 7000000, 6000000],
            TARGET: [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        }
    )

    features = ["Độ tuổi", "Hôn nhân", "Sở hữu BĐS", "Thu nhập"]
    tree = build_tree(data, features)

    print("DỮ LIỆU RỦI RO TÍN DỤNG:\n")
    print(data.to_string(index=False))
    print(f"\nEntropy ban đầu: {entropy(data[TARGET]):.4f}")
    print("\nCÂY QUYẾT ĐỊNH ID3:\n")
    print_tree(tree)


if __name__ == "__main__":
    main()
