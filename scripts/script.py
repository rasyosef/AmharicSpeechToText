def find_average(a_list:list)->float:
    if type(a_list) not in [list, tuple, set]:
        raise TypeError("Argument Type can only be a list, tuple or a set")

    average = sum(a_list) / len(a_list)

    return float(average)


def count_occurence(a_list:list)->dict:
    if type(a_list) not in [list, tuple, set]:
        raise TypeError("Argument Type can only be a list, tuple or a set")

    empty_dict = {}
    for i in a_list:
        if i in empty_dict:
            empty_dict[i] += 1
        else:
            empty_dict[i] = 1
    return empty_dict
    
if __name__ == '__main__':
    data = [0,0,9,0,8,9,0,7]