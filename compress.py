"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> import random
    >>> correct = {}
    >>> lst = []
    >>> for _ in range(1000000):
    ...     integer = random.randint(0, 255)
    ...     lst.append(integer)
    ...     if integer in correct:
    ...         correct[integer] += 1
    ...     else:
    ...         correct[integer] = 1
    >>> d = build_frequency_dict(bytes(lst))
    >>> d == correct
    True
    """
    if text == bytes([]):
        return {}

    byte_freq = {}

    for byte in text:
        if byte in byte_freq:
            byte_freq[byte] += 1
        else:
            byte_freq[byte] = 1
    return byte_freq


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.right
    True
    """
    if len(freq_dict) == 0:
        return HuffmanTree(None)
    elif len(freq_dict) == 1:
        import random
        rand_symbol = random.randint(0, 255)

        symbol = list(freq_dict.keys())[0]
        return HuffmanTree(None, HuffmanTree(symbol), HuffmanTree(rand_symbol))

    nodes = [HuffmanTree(x) for x in list(freq_dict)]
    freq = list(freq_dict.values())

    while len(nodes) != 1:
        min1_freq = min(freq)
        min1_node = nodes[freq.index(min1_freq)]
        freq.remove(min1_freq)
        nodes.remove(min1_node)

        min2_freq = min(freq)
        min2_node = nodes[freq.index(min2_freq)]
        freq.remove(min2_freq)
        nodes.remove(min2_node)

        total_freq = min1_freq + min2_freq

        tree = HuffmanTree(None, min1_node, min2_node)

        nodes.append(tree)
        freq.append(total_freq)

    return nodes[0]


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    >>> tree = HuffmanTree(None, left, right)
    >>> get_codes(tree) == {1: "00", 2: "01", 3: "10", 4: "11"}
    True
    >>> left_right = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    >>> left = HuffmanTree(None, HuffmanTree(7), left_right)
    >>> right_right = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    >>> right = HuffmanTree(None, HuffmanTree(5), right_right)
    >>> tree = HuffmanTree(None, left, right)
    >>> get_codes(tree) == {5: '10', 7: '00', 4: '111', 3: '110', \
    2: '011', 1: '010'}
    True
    >>> freq = {1: 59, 2: 47, 3: 40, 4: 37, 5: 16, 6: 15, 7: 9, 8: 7, 9: 3}
    >>> tree = build_huffman_tree(freq)
    >>> d = get_codes(tree)
    >>> codes = {1: "10", 2: "00", 3: "111", 4: "110", 5: "0111", 6: "0110", \
    7: "0100", 8: "01011", 9: "01010"}
    >>> d == codes
    True
    >>> freq2 = {1: 59, 2: 53, 3: 40, 4: 33, 5: 25, 6: 15, 7: 9, 8: 7, 9: 3}
    >>> tree2 = build_huffman_tree(freq2)
    >>> d2 = get_codes(tree2)
    >>> codes2 = {1: "10", 2: "00", 3: "111", 4: "011", 5: "010", 6: "1100", \
    7: "11010", 8: "110111", 9: "110110"}
    >>> d2 == codes2
    True
    >>> freq3 = {1: 33, 2: 8, 3: 12, 4: 15, 5: 32}
    >>> tree3 = build_huffman_tree(freq3)
    >>> d3 = get_codes(tree3)
    >>> codes3 = {1: "11", 2: "010", 3: "011", 4: "00", 5: "10"}
    >>> d3 == codes3
    True
    >>> freq4 = {97: 8.167, 98: 1.492, 99: 2.782, 100: 4.253, 101: 12.702, \
     102: 2.228, 103: 2.015, 104: 6.094, 105: 6.966, 106: 0.153, \
      107: 0.747, 108: 4.025, 109: 2.406, 110: 6.749, 111: 7.507, \
       112: 1.929, 113: 0.095, 114: 5.987, 115: 6.327, 116: 9.056, \
        117: 2.758, 118: 1.037, 119: 2.365, 120: 0.15, 121: 1.974, \
         122: 0.074}
    >>> tree4 = build_huffman_tree(freq4)
    >>> d4 = get_codes(tree4)
    >>> codes4 = {97: '1110', 98: '110000', 99: '01001', 100: '11111', \
     101: '100', 102: '00100', 103: '110011', 104: '0110', 105: '1011', \
      106: '001011011', 107: '0010111', 108: '11110', 109: '00111', \
       110: '1010', 111: '1101', 112: '110001', 113: '001011001', \
        114: '0101', 115: '0111', 116: '000', 117: '01000', 118: '001010', \
         119: '00110', 120: '001011010', 121: '110010', 122: '001011000'}
    >>> d4 == codes4
    True
    >>> freq5 = {97: 15893, 98: 15130, 99: 22252, 100: 30925, \
     101: 43655, 102: 76925, 103: 78701, 104: 68541, 105: 76030, \
     106: 17641, 107: 61714, 108: 85114, 109: 31278, 110:16029, \
     111: 70114, 112: 50087, 113: 22768, 114: 54435, 115: 59106, \
     116: 31186, 117: 59273, 118: 96581, 119: 25685, 120: 53074, \
     121: 93773, 122: 38733, 65: 91298, 66: 38393, 67: 27410, 68: 76942, \
     69: 71209, 70: 71060, 71: 28487, 72: 35784, 73: 44147, 74: 76974, \
     75: 54513, 76: 82386, 77: 29302, 78: 76353, 79: 40282, 80: 56574, \
     81: 31544, 82: 86082, 83: 90185, 84: 86882, 85: 68426, 86: 63825, \
     87: 73288, 88: 27163, 89: 73996, 90: 39480}
    >>> tree5 = build_huffman_tree(freq5)
    >>> d5 = get_codes(tree5)
    >>> codes5 = {97: '11110001', 98: '11110000', 99: '1011100', \
    100: '1110101', 101: '100110', 102: '01001', 103: '01101', \
    104: '00000', 105: '00111', 106: '0000101', 107: '111011', \
    108: '10000', 109: '1111010', 110: '0000100', 111: '00010', \
    112: '101111', 113: '1011101', 114: '110100', 115: '111000', \
    116: '1111001',117: '111001', 118: '11000', 119: '1100100', \
    120: '110011', 121: '10110', 122: '011001', 65: '10101', \
    66: '011000', 67: '1101100', 68: '01010', 69: '00100', 70: '00011', \
    71: '1101101', 72: '000011', 73: '100111', 74: '01011', 75: '110101', \
    76: '01111', 77: '1110100', 78: '01000', 79: '011101', 80: '110111', \
    81: '1111011', 82: '10001', 83: '10100', 84: '10010', 85: '111111', \
    86: '111110', 87: '00101', 88: '1100101', 89: '00110', 90: '011100'}
    >>> d5 == codes5
    True
    """

    if tree == HuffmanTree(None):
        return {}

    symbol_code = [(tree, "")]

    while True:

        for tree2, string in symbol_code.copy():
            temp = 0

            if not isinstance(tree2, int) and tree2.is_leaf():
                symbol_code.append((tree2.symbol, string))
                temp = 1

            if temp == 0 and not isinstance(tree2, int) and \
                    tree2.right is not None:
                symbol_code.append((tree2.right, string + "1"))

            if temp == 0 and not isinstance(tree2, int) and \
                    tree2.left is not None:
                symbol_code.append((tree2.left, string + "0"))

            if not isinstance(tree2, int):
                symbol_code.remove((tree2, string))

        if all([isinstance(x, int) for x, y in symbol_code]):
            break

    return {x: y for x, y in symbol_code}


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> freq = {1: 59, 2: 47, 3: 40, 4: 37, 5: 16, 6: 15, 7: 9, 8: 7, 9: 3}
    >>> tree = build_huffman_tree(freq)
    >>> number_nodes(tree)
    >>> tree.left.right.left.right.number, tree.left.right.left.number, \
    tree.left.right.right.number, tree.left.right.number, \
    tree.left.number, tree.right.right.number, tree.right.number, \
    tree.number, tree.left.left.number, tree.right.right.right.number
    (0, 1, 2, 3, 4, 5, 6, 7, None, None)
    """

    temp_nodes = [tree]
    all_nodes = []

    while temp_nodes:

        all_nodes.append(temp_nodes.pop())

        if all_nodes[-1].left:
            temp_nodes.append(all_nodes[-1].left)
        if all_nodes[-1].right:
            temp_nodes.append(all_nodes[-1].right)

    counter = 0
    for each in all_nodes[::-1]:
        if not each.is_leaf():
            each.number = counter
            counter += 1


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    >>> freq = {1: 59, 2: 47, 3: 40, 4: 37, 5: 16, 6: 15, 7: 9, 8: 7, 9: 3}
    >>> tree = build_huffman_tree(freq)
    >>> avg_length(tree, freq)
    2.8025751072961373
    >>> freq2 = {1: 59, 2: 53, 3: 40, 4: 33, 5: 25, 6: 15, 7: 9, 8: 7, 9: 3}
    >>> tree2 = build_huffman_tree(freq2)
    >>> avg_length(tree2, freq2)
    2.7991803278688523
    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    """
    if tree == HuffmanTree(None):
        return 0

    codes = get_codes(tree)

    total = 0
    counter = 0

    for symbol, freq in freq_dict.items():
        counter += freq
        total += freq * len(codes[symbol])

    if counter == 0:
        return 0
    else:
        return total / counter


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    lst_codes = []
    code = ""

    for byte in text:
        code += codes[byte]

        while len(code) >= 8:
            lst_codes.append(bits_to_byte(code[:8]))
            code = code[8:]

    lst_codes.append(bits_to_byte(code[:8]))

    return bytes(lst_codes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    temp_nodes = [tree]
    all_nodes = []

    while temp_nodes:

        all_nodes.append(temp_nodes.pop())

        if all_nodes[-1].left:
            temp_nodes.append(all_nodes[-1].left)
        if all_nodes[-1].right:
            temp_nodes.append(all_nodes[-1].right)

    internal_nodes = []
    for each in all_nodes[::-1]:
        if not each.is_leaf():
            internal_nodes.append(each)

    representation = []

    for subtree in internal_nodes:
        if subtree.left.is_leaf():
            representation.append(0)
            representation.append(subtree.left.symbol)
        else:
            representation.append(1)
            representation.append(subtree.left.number)

        if subtree.right.is_leaf():
            representation.append(0)
            representation.append(subtree.right.symbol)
        else:
            representation.append(1)
            representation.append(subtree.right.number)

    return bytes(representation)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> d = {97: 59, 101: 53, 105: 40, 115: 33, 109: 25, 103: 15, 50: 9, \
    116: 7, 110: 3}
    >>> tree = build_huffman_tree(d)
    >>> number_nodes(tree)
    >>> lst = bytes_to_nodes(tree_to_bytes(tree))
    >>> generate_tree_general(lst, -1) == tree
    True
    >>> freq2 = {97: 8.167, 98: 1.492, 99: 2.782, 100: 4.253, 101: 12.702, \
     102: 2.228, 103: 2.015, 104: 6.094, 105: 6.966, 106: 0.153, \
      107: 0.747, 108: 4.025, 109: 2.406, 110: 6.749, 111: 7.507, \
       112: 1.929, 113: 0.095, 114: 5.987, 115: 6.327, 116: 9.056, \
        117: 2.758, 118: 1.037, 119: 2.365, 120: 0.15, 121: 1.974, \
         122: 0.074}
    >>> tree2 = build_huffman_tree(freq2)
    >>> number_nodes(tree2)
    >>> lst2 = bytes_to_nodes(tree_to_bytes(tree2))
    >>> generate_tree_general(lst2, -1) == tree2
    True
    >>> freq3 = {97: 15893, 98: 15130, 99: 22252, 100: 30925, \
     101: 43655, 102: 76925, 103: 78701, 104: 68541, 105: 76030, \
     106: 17641, 107: 61714, 108: 85114, 109: 31278, 110:16029, \
     111: 70114, 112: 50087, 113: 22768, 114: 54435, 115: 59106, \
     116: 31186, 117: 59273, 118: 96581, 119: 25685, 120: 53074, \
     121: 93773, 122: 38733, 65: 91298, 66: 38393, 67: 27410, 68: 76942, \
     69: 71209, 70: 71060, 71: 28487, 72: 35784, 73: 44147, 74: 76974, \
     75: 54513, 76: 82386, 77: 29302, 78: 76353, 79: 40282, 80: 56574, \
     81: 31544, 82: 86082, 83: 90185, 84: 86882, 85: 68426, 86: 63825, \
     87: 73288, 88: 27163, 89: 73996, 90: 39480}
    >>> tree3 = build_huffman_tree(freq3)
    >>> number_nodes(tree3)
    >>> lst3 = bytes_to_nodes(tree_to_bytes(tree3))
    >>> generate_tree_general(lst3, -1) == tree3
    True
    """
    if not node_lst:
        return HuffmanTree(None)

    tree = HuffmanTree(None)
    if node_lst[root_index].l_type == 0:
        tree.left = HuffmanTree(node_lst[root_index].l_data)
    else:
        tree.left = generate_tree_general(node_lst, node_lst[root_index].l_data)

    if node_lst[root_index].r_type == 0:
        tree.right = HuffmanTree(node_lst[root_index].r_data)
    else:
        tree.right = generate_tree_general(node_lst,
                                           node_lst[root_index].r_data)

    return tree


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> d = {97: 59, 101: 53, 105: 40, 115: 33, 109: 25, 103: 15, 50: 9, \
    116: 7, 110: 3}
    >>> tree = build_huffman_tree(d)
    >>> number_nodes(tree)
    >>> lst = bytes_to_nodes(tree_to_bytes(tree))
    >>> generate_tree_postorder(lst, -1) == tree
    True
    >>> freq2 = {97: 8.167, 98: 1.492, 99: 2.782, 100: 4.253, 101: 12.702, \
     102: 2.228, 103: 2.015, 104: 6.094, 105: 6.966, 106: 0.153, \
      107: 0.747, 108: 4.025, 109: 2.406, 110: 6.749, 111: 7.507, \
       112: 1.929, 113: 0.095, 114: 5.987, 115: 6.327, 116: 9.056, \
        117: 2.758, 118: 1.037, 119: 2.365, 120: 0.15, 121: 1.974, \
         122: 0.074}
    >>> tree2 = build_huffman_tree(freq2)
    >>> number_nodes(tree2)
    >>> lst2 = bytes_to_nodes(tree_to_bytes(tree2))
    >>> generate_tree_postorder(lst2, -1) == tree2
    True
    >>> freq3 = {97: 15893, 98: 15130, 99: 22252, 100: 30925, \
     101: 43655, 102: 76925, 103: 78701, 104: 68541, 105: 76030, \
     106: 17641, 107: 61714, 108: 85114, 109: 31278, 110:16029, \
     111: 70114, 112: 50087, 113: 22768, 114: 54435, 115: 59106, \
     116: 31186, 117: 59273, 118: 96581, 119: 25685, 120: 53074, \
     121: 93773, 122: 38733, 65: 91298, 66: 38393, 67: 27410, 68: 76942, \
     69: 71209, 70: 71060, 71: 28487, 72: 35784, 73: 44147, 74: 76974, \
     75: 54513, 76: 82386, 77: 29302, 78: 76353, 79: 40282, 80: 56574, \
     81: 31544, 82: 86082, 83: 90185, 84: 86882, 85: 68426, 86: 63825, \
     87: 73288, 88: 27163, 89: 73996, 90: 39480}
    >>> tree3 = build_huffman_tree(freq3)
    >>> number_nodes(tree3)
    >>> lst3 = bytes_to_nodes(tree_to_bytes(tree3))
    >>> generate_tree_postorder(lst3, -1) == tree3
    True
    >>> import random
    >>> freq4 = {j: random.randint(0, 10**1000) for j in range(0, 256)}
    >>> tree4 = build_huffman_tree(freq4)
    >>> number_nodes(tree4)
    >>> lst4 = bytes_to_nodes(tree_to_bytes(tree4))
    >>> generate_tree_postorder(lst4, -1) == tree4
    True
    """
    if not node_lst:
        return HuffmanTree(None)

    tree = HuffmanTree(None)

    if node_lst[root_index].r_type == 0:
        tree.right = HuffmanTree(node_lst[root_index].r_data)
    else:
        tree.right = generate_tree_postorder(node_lst, root_index-1)

    if node_lst[root_index].l_type == 0:
        tree.left = HuffmanTree(node_lst[root_index].l_data)
    else:
        if node_lst[root_index].r_type == 0:
            tree.left = generate_tree_postorder(node_lst, root_index-1)
        else:
            index = _generate_tree_postorder(node_lst, root_index)

            tree.left = generate_tree_postorder(node_lst, index)

    return tree


def _generate_tree_postorder(node_lst: List[ReadNode], root_index: int) -> int:
    """ Generates the left side of the tree when the right type is 1
    """
    index = root_index-1
    possible_index = root_index-2
    while True:
        if node_lst[index].r_type == 1:
            possible_index -= 1

        if node_lst[index].l_type == 1:
            possible_index -= 1

        index -= 1

        if possible_index == index:
            break

    return index


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    >>> text = b'Hello My name is Kesavar Kabilar'
    >>> tree = build_huffman_tree(build_frequency_dict(text))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(text, get_codes(tree)), len(text)) == text
    True
    """

    original_text = []
    sub_tree = tree

    for byte in text:

        text_code = byte_to_bits(byte)

        for char in text_code:

            if char == "0":
                sub_tree = sub_tree.left
            else:
                sub_tree = sub_tree.right

            if sub_tree.symbol is not None:
                original_text.append(sub_tree.symbol)
                sub_tree = tree

            if len(original_text) == size:
                return bytes(original_text)

    return bytes(original_text)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))

        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    path = list(get_codes(tree).values())
    path = sorted(path, key=len)

    freq_lst = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)

    for i in range(len(freq_lst)):

        subtree = tree

        for branch in path[i]:
            if branch == "0":
                subtree = subtree.left
            else:
                subtree = subtree.right

        subtree.symbol = freq_lst[i][0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
