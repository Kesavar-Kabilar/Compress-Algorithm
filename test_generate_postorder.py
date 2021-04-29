from compress import *
from copy import deepcopy
import pytest


def get_tup(file_name):
    file = open(file_name, 'r')
    final = []
    curr = []
    i = 0
    for line in file:
        if i % 2 == 0:
            curr.append(eval(line))
            i += 1
        else:
            curr.append(eval(line))
            final.append(tuple(curr))
            curr = []
            i += 1
    file.close()
    return final


d = get_tup('final_dict.txt')


def remove_symbols(a):
    final = []
    for n in a:
        if n.l_type == 1:
            n.l_data = 'XX'
        if n.r_type == 1:
            n.r_data = 'XX'
        final.append(n)
    return final


def test_gen_post():
    for h1, d1 in deepcopy(d):
        if len(d1) <= 1:
            continue
        number_nodes(h1)
        a = bytes_to_nodes(tree_to_bytes(h1))
        remove_symbols(a)
        org = deepcopy(a)
        assert generate_tree_postorder(a, -1) == h1


if __name__ == '__main__':
    pytest.main(["test_generate_postorder.py"])
