def normalize_index(items, index):
    if index < 0 or index > len(items):
        raise IndexError(index)
    return items[index]
