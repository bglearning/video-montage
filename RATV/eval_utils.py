def intersection_over_union(predicted: set, target: set) -> float:
    intersection = target.intersection(predicted)
    union = target.union(predicted)
    return len(intersection) / len(union)