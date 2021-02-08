def uniq_list(duplicate_list):
    uniq_list = []
    for elem in duplicate_list:
        if elem not in uniq_list:
            uniq_list.append(elem)
    return uniq_list
