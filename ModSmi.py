import os
import re
import time
import random
import itertools

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

limit_ = 200

tokens = {
    "<",
    ">",
    "#",
    "%",
    ")",
    "(",
    "+",
    "-",
    "/",
    ".",
    "1",
    "0",
    "3",
    "2",
    "5",
    "4",
    "7",
    "6",
    "9",
    "8",
    "=",
    "A",
    "@",
    "C",
    "B",
    "F",
    "I",
    "H",
    "O",
    "N",
    "P",
    "S",
    "[",
    "]",
    "\\",
    "c",
    "e",
    "i",
    "l",
    "o",
    "n",
    "p",
    "s",
    "r",
}
# token_set = {'#', '%', '(', ')', '+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C',
#              'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', ']', 'c', 'e', 'i', 'l', 'n', 'o', 'p', 'r', 's'}

bond_tokens = {"#", ".", "=", "[", "]"}

num_tokens = [str(i) for i in range(1, 10)]


# def chk_token(filename_):
#     path_ = '/home/yongbeom/research/gen_smiles'
#     if os.path.exists(f'{path_}/token.list'):
#         with open(f'{path_}/token.list') as _if:
#             token = _if.read().split(',')
#             # print(token)
#             token = set(token)
#     else:
#         token = {}

#     with open(filename_) as f:
#         for l in f:
#             smiles, cid = l.strip().split()
#             print(smiles, cid)
#             if len(set(list(smiles)) - token) > 0:
#                 token.update(set(list(smiles)) - token)

#     token = sorted(token)

#     with open(f'{path_}/token.list', 'w') as of:
#         of.write(','.join(token))


def set_avoid_ring(_smiles):
    avoid_ring = []
    ring_tmp = set(re.findall(r"\d", _smiles))
    for j in ring_tmp:
        tmp = [i for i, val in enumerate(_smiles) if val == j]
        while tmp:
            avoid_ring += [j for j in range(tmp.pop(0), tmp.pop(0) + 1)]
    return set(avoid_ring)


def prepare_rigid_crossover(_smiles, side, consider_ring=True, _minimum_len=4):
    """ 1 point crossover
    :param _smiles: SMILES (str)
    :param side: Left SMILES or Right SMILES ['L'|'R'] (str)
    :param consider_ring: consider about avoiding ring (bool)
    :param _minimum_len: minimum cut size (int)
    :return:
    """

    if side not in ["L", "R"]:
        raise Exception("You must choice in L(Left) or R(Right)")

    _smiles_len = len(_smiles)
    _smi = None

    if consider_ring:  # ring 부분을 피해서 자르도록 고려.
        avoid_ring_list = set_avoid_ring(_smiles)

    p = 0
    _start = None
    _end = None
    _gate = False
    while not _gate:  # 원하는 형태의 smiles piece가 나올 때 까지 반복
        if p == limit_:
            raise ValueError(f"main_gate fail ({side}): {_smiles}")

        if consider_ring:
            j = 0
            ring_gate = False
            if side == "L":
                while not ring_gate:
                    if j == 30:
                        raise ValueError(f"ring_gate fail (L): {_smiles}")
                    _end = np.random.randint(_minimum_len, _smiles_len + 1)
                    if _end not in avoid_ring_list:
                        ring_gate = True
                    j += 1
            elif side == "R":
                while not ring_gate:
                    if j == 30:
                        raise ValueError(f"ring_gate fail (R): {_smiles}")
                    _start = np.random.randint(0, _smiles_len - _minimum_len)
                    if _start not in avoid_ring_list:
                        ring_gate = True
                    j += 1
            _smi = _smiles[_start:_end]
        else:
            if side == "L":
                _end = np.random.randint(_minimum_len, _smiles_len)
            elif side == "R":
                _start = np.random.randint(0, _smiles_len - _minimum_len)

            _smi = _smiles[_start:_end]
            chk_ring = re.findall(r"\d", _smi)
            i = 0
            for i in set(chk_ring):
                list_ring = [_ for _, val in enumerate(_smi) if val == i]
                if (len(list_ring) % 2) is 1:
                    b = random.sample(list_ring, 1)
                    _smi = _smi[: b[0]] + _smi[b[0] + 1 :]
                    # print(f'@ {_smi} // {_smiles}')

        p += 1

        if "." in _smi:  # 이온은 패스한다.
            continue

        n_chk = 0
        for j in _smi:  # [] 닫혀 있는지 확인.
            if j == "[":
                n_chk += 1
            if j == "]":
                n_chk -= 1
        if n_chk == 0:
            _gate = True

    return _smi


def random_slicing_smiles(_smiles, consider_ring=True, _minimum_len=4, len_factor=1):
    """
    """

    _smiles_len = len(_smiles)
    max_limit_len = int(np.ceil(_smiles_len * len_factor))
    _smi = None
    n_chk = 0

    avoid_ring = []
    # print(f'#   0 {idx:6} ring_numbers        ', end='\r')
    if consider_ring:  # ring 부분을 피해서 자르도록 고려.
        ring_numbers = set(re.findall(r"\d", _smiles))
        for j in ring_numbers:
            tmp = [i for i, val in enumerate(_smiles) if val == j]
            # ring_dict[j] = [i for i, val in enumerate(_smiles) if val == j]
            while tmp:
                avoid_ring += [j for j in range(tmp.pop(0), tmp.pop(0) + 1)]

        avoid_ring = set(avoid_ring)

    n_limit = 0
    main_gate = False
    # print(f'#   0 {idx:6} main_gate        ', end='\r')
    while not main_gate:  # 원하는 형태의 smiles piece가 나올 때 까지 반복
        # avoid_ring = set(double_list_to_(ring_dict.values()))
        if n_limit == round(limit_ * 0.5):
            # raise PermissionError(f'main_gate fail (M): {_smiles}')
            raise ValueError(f"main_gate fail (M): {_smiles}")

        if consider_ring:
            j = 0
            ring_gate = False
            # print(f'#   0 {idx:6} ring_gate        ', end='\r')
            while not ring_gate:  # random slicing
                if j == limit_:
                    raise ValueError(f"ring_gate fail (M): {_smiles}")

                rnd_len = np.random.randint(_minimum_len, max_limit_len)
                _start = np.random.randint(0, _smiles_len - rnd_len)
                _end = _start + rnd_len

                j += 1
                # print(f'#   0 {idx:6} ring_gate@   {j:4}', end='\r')

                if (_start not in avoid_ring) and (_end not in avoid_ring):
                    _smi = _smiles[_start:_end]
                    if "." in _smi:  # 이온은 패스한다.
                        continue

                    n_chk = 0
                    for k in _smi:  # [] 닫혀 있는지 확인.
                        if k == "[":
                            n_chk += 1
                        if k == "]":
                            n_chk -= 1
                        if n_chk < 0:
                            # print(_smi)
                            break

                    # print(_start and _end)
                    # print(f'start {_start} end {_end}, {avoid_ring}')
                    ring_gate = True

                ### ______1_____ 1 (X); 1 _______ 1 (O) 아래의 전략은 링번호가 랜덤범위에 속하지 않도록 한다.
                # if _end not in avoid_ring:
                #     ring_gate = True
                # tmp_chk_ring = [_ for _ in range(_start, _end + 1)]
                # if j > round(limit_*0.7):
                #     if j % (limit_*0.1) == 0:
                #         i += 1
                #         avoid_ring -= set(ring_dict[str(i)])
                #     if not any(x in avoid_ring for x in tmp_chk_ring):
                #         ring_gate = True
                # else:
                #     if not any(x in avoid_ring for x in tmp_chk_ring):
                #         ring_gate = True

        else:
            rnd_len = np.random.randint(_minimum_len, max_limit_len)
            _start = np.random.randint(0, _smiles_len - rnd_len)
            _end = _start + rnd_len

            _smi = _smiles[_start:_end]

        n_limit += 1

        if n_chk == 0:
            main_gate = True
        # print(f'#   0 {idx:6} ring_gate  {n_chk:2}{n_limit:4}', end='\r')

    return _smi


def prepare_flexible_crossover(_smiles, side, consider_ring=True, _minimum_len=4, len_factor=1):
    """
    random slicing SMILES
    :param _smiles: SMILES (str)
    :param side: Left SMILES or Right SMILES ['L'|'R'] (str)
    :param consider_ring: consider about avoiding ring (bool)
    :param _minimum_len: minimum cut size (int)
    :param len_factor: If random slice case, it adjusts a possible max length.
    :return:
    """

    if side not in ["L", "R"]:
        raise Exception("You must choice in L(Left) or R(Right)")

    _smi = None

    _smiles = random_slicing_smiles(_smiles, consider_ring, _minimum_len, len_factor)

    if side == "L":
        _smi = l_checker(_smiles)
    elif side == "R":
        _smi = r_checker(_smiles)

    return _smi


def prepare_relaxed_crossover(_smiles, _minimum_len=4, len_factor=1):
    """
    ring 자르는 것을 고려하지 않고, 막 슬라이싱 한 후 숫자만 맞춰 줌. #
    :param _smiles: SMILES (str)
    :param _minimum_len: minimum cut size (int)
    :param len_factor: If random slice case, it adjusts a possible max length.
    :return:
    """

    _smiles_len = len(_smiles)
    max_limit_len = int(np.ceil(_smiles_len * len_factor))
    _start = None
    _end = None
    _smi = None

    p = 0
    main_gate = False
    while not main_gate:  # 원하는 형태의 smiles piece가 나올 때 까지 반복
        if p == limit_:
            # raise PermissionError(f'main_gate fail (M): {_smiles}')
            raise ValueError(f"main_gate fail (M): {_smiles}")

        rnd_len = np.random.randint(_minimum_len, max_limit_len)
        _start = np.random.randint(0, _smiles_len - rnd_len)
        _end = _start + rnd_len

        p += 1

        _smi = _smiles[_start:_end]
        # if p > 10:
        # print(_smi)

        if "." in _smi:  # 이온은 패스한다.
            continue

        n_chk = 0
        for k in _smi:  # [] 닫혀 있는지 확인.
            if k == "[":
                n_chk += 1
            if k == "]":
                n_chk -= 1

        chk_ring = re.findall(r"\d", _smi)
        i = 0
        for i in set(chk_ring):
            list_ring = [_ for _, val in enumerate(_smi) if val == i]
            if (len(list_ring) % 2) is 1:
                b = random.sample(list_ring, 1)
                _smi = _smi[: b[0]] + _smi[b[0] + 1 :]
                # print(f'@ {_smi} // {_smiles}')

        if n_chk == 0:
            main_gate = True

    return _smi


def chk_branch(_smi, side=None):

    if side not in ["L", "R", None]:
        raise Exception("You must choice in L(Left) or R(Right)")

    branch_list = []
    n_branch = 0
    min_branch = 0
    for i, b in enumerate(_smi):  # () 닫혀 있는지 확인.
        if b is "(":
            n_branch += 1  # 0 == (
        if b in ")":
            n_branch -= 1  # 1 == )
        if side is "L":
            if n_branch < min_branch:
                min_branch = n_branch
                branch_list.append(i)
        elif side is "R":  # track max_value
            if n_branch > min_branch:
                min_branch = n_branch
                branch_list.append(i)
    if side is None:
        return n_branch
    return np.asarray(branch_list), min_branch


def l_checker(_smi):
    """
    합치기 전의 왼쪽 스마일에 대한 정보를 전처리한다.
    특히, ) 과 같이 닫혀만 있는 것을 처리한다.
    알려진 문제점:
    @Before =O)NC(C
    @After  O)NC(C
    @Before =CC=CC
    @After  CC=CC
    다음 과 같이 이중결합을 제거함. 제거하지 않으면 invalid SMILES 가 됨.
    :param _smi:
    :return:
    """

    side = "L"

    tmp = re.findall(r"[^(]", _smi)[0]
    # print(f'#   0 {idx:6} l_tmp         ', end='\r')
    while tmp in [")", "=", "#"]:
        _smi = _smi[: _smi.find(tmp)] + _smi[_smi.find(tmp) + 1 :]
        tmp = re.findall(r"[^(]", _smi)[0]

    branch_list, min_branch = chk_branch(_smi, side)

    if min_branch < 0:  # resolving over closed branches.
        # print(f'BEFORE/ {min_branch}, _smi: {_smi}')
        # print(f'#   0 {idx:6} l_branch         ', end='\r')
        while min_branch != 0:
            n = np.random.rand()
            if n > 0.6:  # relax insert "("
                close_branch = [i for i, e in enumerate(_smi) if e in [")", "+", "-", "]"]]

                p = 0
                b = True
                # print(f'#   0 {idx:6} l_branch @@      ', end='\r')
                while b in close_branch:
                    b = np.random.randint(0, branch_list[-1])
                    p += 1
                    if p > 50:
                        # print(f'#   0 {p:6} l_branch @@ {_smi}', end='\r')
                        break

                _smi = _smi[:b] + "(" + _smi[b:]

            else:  # removing closed branch.
                b = np.random.choice(branch_list, 1)[0]
                _smi = _smi[:b] + _smi[b + 1 :]

            # print(f'AFTER/ remove: {min_branch}, _smi: {_smi}')
            branch_list, min_branch = chk_branch(_smi, side)

    return _smi


def r_checker(_smi):
    """
    합치기 전의 왼쪽 스마일에 대한 정보를 전처리한다.
    특히, ( 과 같이 열려만 있는 것을 처리한다.
    SMILES를 반전 시켜 적용한다.
    :param _smi:
    :return:
    """

    side = "R"

    _smi = _smi[::-1]

    tmp = re.findall(r"[^)]", _smi)[0]
    # print(f'#   0 {idx:6} r_tmp           ', end='\r')
    while tmp in ["(", "=", "#"]:
        _smi = _smi[: _smi.find(tmp)] + _smi[_smi.find(tmp) + 1 :]
        tmp = re.findall(r"[^)]", _smi)[0]

    branch_list, min_branch = chk_branch(_smi, side)

    if min_branch > 0:
        # print(f'BEFORE/ {min_branch}, _smi: {_smi[::-1]}')
        # print(f'#   0 {idx:6} r_branch         ', end='\r')
        while min_branch != 0:
            n = np.random.rand()
            if n > 0.6:  # relax insert ")"
                open_branch = [
                    i for i, e in enumerate(_smi) if e in ["(", "+", "-", "[", "]", "=", "#"]
                ]

                p = 0
                b = True
                # print(f'#   0 {idx:6} r_branch @@      ', end='\r')
                while b in open_branch:
                    b = np.random.randint(0, branch_list[-1])
                    p += 1
                    if p > 50:
                        # print(f'#   0 {p:6} l_branch @@ {_smi}', end='\r')
                        break

                _smi = _smi[:b] + ")" + _smi[b:]

            else:  # removing opened branch.
                b = np.random.choice(branch_list, 1)[0]
                _smi = _smi[:b] + _smi[b + 1 :]

            # print(f'AFTER/ remove: {min_branch}, _smi: {_smi[::-1]}')
            branch_list, min_branch = chk_branch(_smi, side)

    return _smi[::-1]


def get_open_branch(_smi):
    return [i for i, e in enumerate(_smi) if e == "("]


def get_close_branch(_smi):
    return [i for i, e in enumerate(_smi) if e == ")"]


def tight_rm_branch(_smi_l, _smi_r):
    # tmp = time.time()

    _new_smi = _smi_l + _smi_r

    open_branch = get_open_branch(_new_smi)
    close_branch = get_close_branch(_new_smi)

    b = None
    n_branch = chk_branch(_new_smi)

    q = len(_smi_l)
    while n_branch > 0:  # over opened-branch
        _smi_l_open_branch = get_open_branch(_smi_l)
        _smi_r_open_branch = get_open_branch(_smi_r)
        open_branch = get_open_branch(_smi_l + _smi_r)
        avoid_tokens = [
            i
            for i, e in enumerate(_smi_l + _smi_r)
            if e in ["=", "#", "@", "1", "2", "3", "4", "5", "6", "7", "8"]
        ]

        if len(_smi_r_open_branch) == 0:  # open branch 가 없을 경우
            _smi_r_open_branch.append(len(_smi_r))
        if len(_smi_l_open_branch) == 0:
            _smi_l_open_branch.append(0)

        n = np.random.rand()  # 임의적으로 close branch 를 추가하거나 제거한다.
        if n > 0.5:  # 추가
            branch_gate = False
            j = 0
            while not branch_gate:  # Ring 부분을 피해서 자름
                if j == limit_:
                    raise ValueError
                b = np.random.randint(_smi_l_open_branch[-1] + 1, _smi_r_open_branch[-1] + q)
                j += 1
                if b not in avoid_tokens:
                    branch_gate = True
            n_branch -= 1
            if b <= len(_smi_l):  # SMILES 길이를 고려하여 자른다. 좌측 SMILES의 open branch를 cut!
                _smi_l = _smi_l[:b] + ")" + _smi_l[b:]
                q += 1
            else:  # 좌측 SMILES 길이를 제외한 수가 우측 SMILES 문자의 위치를 의미한다.
                b -= len(_smi_l)
                _smi_r = _smi_r[:b] + ")" + _smi_r[b:]
        else:  # 제거
            b = _smi_l_open_branch[-1]  # (Random으로도 가능함. 과한 부분만 Cut!)
            n_branch -= 1
            q -= 1
            _smi_l = _smi_l[:b] + _smi_l[b + 1 :]

    while n_branch < 0:  # over closed-branch
        _smi_l_close_branch = get_close_branch(_smi_l)
        _smi_r_close_branch = get_close_branch(_smi_r)
        close_branch = get_close_branch(_smi_l + _smi_r)
        avoid_tokens = [
            i
            for i, e in enumerate(_smi_l + _smi_r)
            if e in ["=", "#", "@", "1", "2", "3", "4", "5", "6", "7", "8"]
        ]

        if len(_smi_r_close_branch) == 0:
            _smi_r_close_branch.append(len(_smi_r))
        if len(_smi_l_close_branch) == 0:
            _smi_l_close_branch.append(0)

        n = np.random.rand()
        if n > 0.5:
            branch_gate = False
            j = 0
            while not branch_gate:  # Ring 부분을 피해서 자름
                b = np.random.randint(_smi_l_close_branch[-1] + 1, _smi_r_close_branch[0] + q)
                j += 1
                if b not in (close_branch + avoid_tokens):
                    branch_gate = True
                if j == limit_:
                    raise ValueError
            n_branch += 1
            if b < len(_smi_l):
                _smi_l = _smi_l[:b] + "(" + _smi_l[b:]
                q += 1
            else:
                b -= len(_smi_l)
                _smi_r = _smi_r[:b] + "(" + _smi_r[b:]
        else:
            b = _smi_r_close_branch[0]
            n_branch += 1
            # print(f'{_smi_r[b]}')
            _smi_r = _smi_r[:b] + _smi_r[b + 1 :]

    # time_.append(time.time() - tmp)

    return _smi_l + _smi_r


def replace_atom(_smi):

    #                    C  Si/ B  N  P / O  S / F  Cl  Br  I
    replace_atom_list = [6, 14, 5, 7, 15, 8, 16, 9, 17, 35, 53]
    #                         C  N  P / O  S
    replace_arom_atom_list = [6, 7, 15, 8, 16]

    # print(f"before: {_smi}")

    mol_ = Chem.MolFromSmiles(_smi)
    max_len = mol_.GetNumAtoms()

    mw = Chem.RWMol(mol_)
    # Chem.SanitizeMol(mw)

    p = 0
    gate_ = False
    while not gate_:
        if p is 30:
            # raise Exception
            raise PermissionError

        rnd_atom = np.random.randint(0, max_len)

        valence = mw.GetAtomWithIdx(rnd_atom).GetExplicitValence()
        if mw.GetAtomWithIdx(rnd_atom).GetIsAromatic():
            if valence is 3:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_arom_atom_list[np.random.randint(0, 3)]))
            elif valence is 2:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_arom_atom_list[np.random.randint(1, 5)]))
            else:
                continue
            mw.GetAtomWithIdx(rnd_atom).SetIsAromatic(True)
        else:
            if valence is 4:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_atom_list[np.random.randint(0, 2)]))
            elif valence is 3:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_atom_list[np.random.randint(0, 5)]))
            elif valence is 2:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_atom_list[np.random.randint(0, 7)]))
            elif valence is 1:
                mw.ReplaceAtom(rnd_atom, Chem.Atom(replace_atom_list[np.random.randint(0, 11)]))

        p += 1
        # print(f"after: {Chem.MolToSmiles(mw)}")
        try:
            Chem.SanitizeMol(mw)
            gate_ = True
        except Chem.rdchem.KekulizeException:
            # print(f"{_smi} {Chem.MolToSmiles(mw, kekuleSmiles=False)} {Chem.MolToSmiles(mol_, kekuleSmiles=False)}")
            # raise Exception
            pass

    Chem.Kekulize(mw)
    # print(f"after_San: {Chem.MolToSmiles(mw)}")

    return Chem.MolToSmiles(mw, kekuleSmiles=True), mw


def delete_atom(_smi):
    """
    Aromatic ring 을 제외하고 삭제함.
    :param _smi:
    :return:
    """
    max_len = len(_smi)
    mol_ = False
    _new_smi = None

    p = 0
    while not mol_:
        p += 1
        if p is 30:
            # raise Exception
            raise PermissionError

        rnd_insert = np.random.randint(max_len)
        _new_smi = _smi[:rnd_insert] + _smi[rnd_insert + 1 :]
        mol_ = Chem.MolFromSmiles(_new_smi)

    # MW type delete; It's has problem, Because ion is generated.
    # mol_ = Chem.MolFromSmiles(_smi)
    # max_len = mol_.GetNumAtoms()

    # mw = Chem.RWMol(mol_)

    # p = 0
    # gate_ = False
    # while not gate_:
    #     if p is 30:
    #         raise Exception
    #     rnd_atom = np.random.randint(0, max_len)
    #     if not mw.GetAtomWithIdx(rnd_atom).GetIsAromatic():
    #         mw.RemoveAtom(rnd_atom)
    #         gate_ = True
    #     p += 1

    # Chem.SanitizeMol(mw)

    # return Chem.MolToSmiles(mw, kekuleSmiles=True), mw
    return _new_smi, mol_


def add_atom(_smi):
    list_atom = ["C", "Si", "B", "N", "P", "O", "S", "Cl", "Br"]
    # avoid_ring = [i for i, j in enumerate(_smi) if j.islower() or j.isdigit()]
    # rnd_insert = np.random.randint(max_len)

    max_len = len(_smi)
    mol_ = False
    _new_smi = None

    p = 0
    while not mol_:
        p += 1
        if p is 30:
            # raise Exception
            raise PermissionError

        rnd_insert = np.random.randint(max_len)
        _new_smi = _smi[:rnd_insert] + random.sample(list_atom, 1)[0] + _smi[rnd_insert:]
        mol_ = Chem.MolFromSmiles(_new_smi)

    return _new_smi, mol_


# def flexible_crossover(_smi_l, _smi_r):
#     tmp = time.time()

#     _new_smi = _smi_l + _smi_r
#     # print("@", _new_smi)

#     n_branch = 0
#     for b in enumerate(_new_smi):  # () 닫혀 있는지 확인.
#         if b is "(":
#             n_branch += 1  # 0 == (
#         if b in ")":
#             n_branch -= 1  # 1 == )

#     if n_branch < 0:
#         _new_smi = l_checker(_new_smi)
#         # print("@-", _new_smi)

#     if n_branch > 0:
#         _new_smi = r_checker(_new_smi)
#         # print("@+", _new_smi)

#     time_.append(time.time() - tmp)

#     return _smi_l + _smi_r


if __name__ == "__main__":

    from multiprocessing import Pool
    import datetime

    idx = 0
    time_ = []
    smiles = []
    cids = []
    with open(
        # "/home/yongbeom/research/gen_smiles/Data/all_pubchem_prob_0.01.smi"
        # "/home/yongbeom/research/gen_smiles/Data/chemvae_test_sample1000.csv"
        "/home/yongbeom/Data/sampled_ZINC_0.001.smi"
        # "./sampled_ZINC_100000.smi"
    ) as inf:
        a = 0
        for l in inf:
            if a == 0:
                a += 1
                continue
            # smile, cid = l.strip().split()
            smile = l.split(" ")[0]
            # print(smile)
            # mol = Chem.MolFromSmiles(smile)
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            smiles.append(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False))
            # print("@", Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False))
            # cids.append(cid)

    def func(*args):
        return prepare_rigid_crossover(*args)  # 1
        # return prepare_rigid_crossover(*args, False)  # 2
        # return prepare_flexible_crossover(*args)  # 3

    # smi1, smi2 = random.sample(smiles, 2)
    # smi = random.sample(smiles)
    # mutaion(smi)

    # l_smi = func(smi1, "L")
    # r_smi = func(smi2, "R")

    # flexible_crossover(None, None)

    # def try_mutaion(smi):
    #     # return replace_atom(smi)[0]  # 91.79% at ZINC data
    #     # return delete_atom(smi)[0]  # 98.15% at ZINC data
    #     return add_atom(smi)[0]  # 100.0% at ZINC data

    # success = 0
    # fail = 0

    # for smi in smiles:
    #     try:
    #         new_mol = Chem.MolFromSmiles(try_mutaion(smi))
    #         if new_mol:
    #             success += 1
    #         else:
    #             print("@@@@@@@@@@@")
    #             raise Exception
    #     except:
    #         # print(smi)
    #         fail += 1
    # s = time.time()

    # with Pool(2) as pool:
    #     imap_it = pool.map(try_mutaion, smiles)
    #     for smi in imap_it:
    #         try:
    #             new_mol = Chem.MolFromSmiles(smi)
    #             if new_mol:
    #                 success += 1
    #             else:
    #                 raise Exception
    #         except:
    #             fail += 1

    # print(f"Success rate: {success/(success+fail)*100}%")
    # print(f"time: {time.time() - s} s")
    # print(prepare_flexible_crossover(smi, 'L'))

    # MAIN LOOP ########################################################
    # log_f = open(f"gen_smiles.log", "w")
    
    print(f"Start at {datetime.datetime.now()}")
    s = time.time()

    niter = 1000000  # 2000000 55min

    # success_f = open(f"success_smiles.test.txt", "w")
    # fail_f = open(f"fail_smiles.txt", "w")
    # fail_f.write(f"L,R\n")

    # global n_valid
    # global n_fail

    n_valid = 0
    n_fail = 0

    def sum_smiles(smi1, smi2):
        l_smi = None
        r_smi = None
        try:
            l_smi = func(smi1, "L", False)
            r_smi = func(smi2, "R", False)
        except (IndexError, ValueError):
            # fail_f.write(f"{l_smi},{r_smi},piece\n")
            # print(f"{l_smi},{r_smi} - piece")
            # n_fail += 1
            raise PermissionError

        k = 0
        mol = False
        while not mol:
            # try:
            # Simple crossover; 단지 자르기만 한다. L: [:rand], R: [rand:] -------------------
            # new_smi = l_smi + r_smi
            # Branch control; ()를 고려한다. L: [:rand], R: [rand:] -------------------
            new_smi = tight_rm_branch(l_smi, r_smi)
            # except (IndexError, ValueError):
                # raise PermissionError
            # Consider ring; 고리를 고려하여 자른다. L: [:rand], R: [rand:] -------------------
            # new_smi = flexible_crossover(l_smi, r_smi)
            # print(f'{smi1} + {smi2}')
            # print(f'{n_branch}, SMILES: {new_smi}')
            mol = Chem.MolFromSmiles(new_smi)
            k += 1
            if k == 30:
                raise PermissionError
                # break
        return mol

    for idx in range(niter):
    # def random_sample(_):
    #     return random.sample(smiles, 2)
    # with Pool(6) as pool:
    #     imap_it = pool.map(random_sample, [i for i in range(niter)])
        # for smis in imap_it:
        # print(smis)
        new_smi = None
        mol = None
        smi1, smi2 = random.sample(smiles, 2)
        try:
            mol = sum_smiles(smi1, smi2)
        except (PermissionError, IndexError, ValueError):
            try:
                mol = sum_smiles(smi2, smi1)
            except (PermissionError, IndexError, ValueError):
            # except PermissionError:
                n_fail += 1
                continue
        
        if mol:
            n_valid += 1
            # print("#", new_smi)
            # success_f.write(f"{new_smi}\n")
        else:
            n_fail += 1
                # fail_f.write(f"{new_smi}\n")

            if idx % 100000 == 0:
                print(f"Success rate: {n_valid/(n_valid + n_fail)*100}%")
    print(f"Final Success rate: {n_valid/(n_valid + n_fail)*100}%")
            # log_f.write(f"Success rate: {n_valid/(n_valid + n_fail)*100}%\n")

    #     # print(f'#   0 {idx:6}                  ', end='\r')
    #     # print(f'##  1 {idx:6}                  ', end='\r')

    # ---------------------------------------------------------------------- 

    # success_f.close()
    # fail_f.close()

    # print("Number of valid molecules: %d" % (n_valid))
    # print(f"Success rate(total): {n_valid / (n_valid + n_fail) * 100}%")
    # log_f.write("Number of valid molecules: %d\n" % (n_valid))
    # log_f.write(f"Success rate(total): {n_valid / (n_valid + n_fail) * 100}%\n")
    # log_f.write(f"Generation time: {(time.time() - s)/60}min\n")
    print(f"Cost time: {(time.time() - s)/60}min\n")
    print(f"End at {datetime.datetime.now()}")
    # log_f.close()

    # time_ = np.asarray(time_)
    # print(time_.mean(), "Avg Time")
    # # # np.save("time_check_rm_ver1", time_)
