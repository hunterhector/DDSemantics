def load_index(index_file):
    """Read the index file"""
    index_dict = {}
    with open(index_file) as f:
        for line in f:
            title, path = line.strip().split()
            index_dict[title] = path
    return index_dict
