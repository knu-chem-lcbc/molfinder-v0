"""
MolFinder ver 0.1
csa_yongbeom2 에서 local minimization 추가
bad replace 조건 추가
"""
import sys, os
import time
import argparse

# from numba import jit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import QED, AllChem

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from sascorer import calculateScore
import ModSmi


parser = argparse.ArgumentParser(description="?")
parser.add_argument("-N", "--nbank", metavar="N", type=int, default=None, help="Bank Size")
parser.add_argument("-r", "--rseed", metavar="N", type=int, default=None, help="random seed")
parser.add_argument("-n", "--nconvergent", metavar="N", type=int, required=True, help="# of convergent round")
parser.add_argument("-v", "--value", metavar="N", type=int, required=True, help="target value ex) Davg/3")
parser.add_argument("-c", "--coef", metavar="coef. of QED", type=float, default=0.83, help="QED")
parser.add_argument("-dc", "--distancecoef", metavar="coef. of distance", type=float, default=1, help="Control Dcut")
parser.add_argument("-t", "--target", metavar="SMILES", type=str, default=None, help="target_moleclue SMILES")
args = parser.parse_args()

if args.target:
    # target_fps = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(args.target), 2)
    target_fps = Chem.RDKFingerprint(Chem.MolFromSmiles(args.target))
    sim_coef = args.coef

    def obj_eq(x):
        compare_target = []
        for mol_ in x[:, 1]:
            # fps_ = AllChem.GetMorganFingerprint(mol_, 2)
            fps_ = Chem.RDKFingerprint(mol_)
            compare_target.append(TanimotoSimilarity(fps_, target_fps))
        compare_target = np.asarray(compare_target)

        return sim_coef * compare_target + (1 - sim_coef) * x[:, 4]
else:
    qed_coef = args.coef
    sas_coef = 1 - args.coef
    obj_eq = lambda x: qed_coef * x[:, 4] - sas_coef * x[:, 3]


class ChkTime:
    def __init__(self, name=None):
        self.t = time.time()
        self.name = name

    def get(self):
        # print(f'{self.name} {time.time() - self.t:.6f} sec')

        return (time.time() - self.t) / 60


def cal_avg_dist(solutions):

    dist_sum = 0
    min_dist = 10
    max_dist = 0
    n_ = len(solutions)

    for i in range(n_ - 1):
        for j in range(i + 1, n_):
            # fps1 = AllChem.GetMorganFingerprint(solutions[i, 1], 2)
            fps1 = Chem.RDKFingerprint(solutions[i, 1])
            # fps2 = AllChem.GetMorganFingerprint(solutions[j, 1], 2)
            fps2 = Chem.RDKFingerprint(solutions[j, 1])
            dist = TanimotoSimilarity(fps1, fps2)
            dist_sum += dist
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return dist_sum / (n_ * (n_ - 1) / 2)  # , min_dist, max_dist


def cal_rnd_avg_dist(solutions, nrnd=400000):

    dist_sum = 0
    min_dist = 10
    max_dist = 0
    tmp_chk = 0

    start_chk = time.time()
    for _ in range(nrnd):  # 300000 반복 이상이여야 .3자리까지 확보가 됨.

        if _ == 0:
            tmp_chk = start_chk

        mol1, mol2 = np.random.choice(solutions[:, 1], size=2)
        # fps1 = AllChem.GetMorganFingerprint(mol1, 2)
        fps1 = Chem.RDKFingerprint(mol1)
        # fps2 = AllChem.GetMorganFingerprint(mol2, 2)
        fps2 = Chem.RDKFingerprint(mol2)
        dist = TanimotoSimilarity(fps1, fps2)
        dist_sum += dist

        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
        if _ % int(nrnd / 10) == 0:
            print(f"{_/nrnd*100:.1f}% complete {(time.time() - tmp_chk)/60} min/10%\r")
            tmp_chk = time.time()

    print(f"calc. Dist total {(time.time() - start_chk)/60} min")

    return dist_sum / nrnd  # , min_dist, max_dist


def cal_array_dist(solutions1, solutions2):
    """
    numpy
    :param solutions1:
    :param solutions2:
    :return:
    """

    n1 = len(solutions1)
    n2 = len(solutions2)
    # n2 = solutions2.shape[0]
    dist = np.zeros([n1, n2])

    for i in range(n1):
        for j in range(n2):
            # fps1 = AllChem.GetMorganFingerprint(solutions1[i, 1], 2)
            fps1 = Chem.RDKFingerprint(solutions1[i, 1])
            # fps2 = AllChem.GetMorganFingerprint(solutions2[j, 1], 2)
            fps2 = Chem.RDKFingerprint(solutions2[j, 1])
            dist[n1, n2] = TanimotoSimilarity(fps1, fps2)

    return dist


def init_bank(file_name, n1=None, n2=None, rseed=None):
    """
    기존의 파일 포맷에서 mol format을 추가하고, random 하게 섞는다.
    :param file_name: 파일 명
    :param n1: nbank
    :param n2: 불러 올 smiles 수
    :param rseed: random seed 값 (int)
    :return:
    """

    np.random.seed(rseed)

    df = pd.read_csv(file_name)
    df = df[:n2].values

    shuffled_index = np.random.permutation(len(df))
    tmp_bank = df[shuffled_index][:n1]
    df = pd.DataFrame(tmp_bank, columns=["SMILES", "SAS", "QED", "Len", "Ring"])
    df.to_csv(f"init_bank_{R_d:.5f}_{args.coef:.3f}.csv", index=False)

    # print(smiles.dtype)
    # print(f'origin: {sys.getsizeof(smiles)}')
    # bank = np.empty([tmp_bank.shape[0], tmp_bank.shape[1] + 1], dtype=object)
    bank = np.empty([tmp_bank.shape[0], tmp_bank.shape[1]], dtype=object)
    # print(f'bank: {sys.getsizeof(bank)}')
    # bvar = np.empty([tmp_bank.shape[0], tmp_bank.shape[1]-1])  #, dtype=np.float16)
    # print(f'bvar: {sys.getsizeof(bvar)}')
    bank[:, 0] = tmp_bank[:, 0]  # SMILES
    bank[:, 2] = True  # usable label True
    bank[:, 3:] = tmp_bank[:, 1:-2]
    # bvar[:] = tmp_bank[:, 1:]

    if args.target:
        _mol = Chem.MolFromSmiles(args.target)
        bank[0, 0] = args.target  # SMILES
        bank[0, 1] = True  # usable label True
        bank[0, 2] = _mol
        bank[0, 3] = calculateScore(_mol)
        bank[0, 4] = QED.default(_mol)

    for i, j in enumerate(bank[:, 0]):
        mol = Chem.MolFromSmiles(j)
        bank[i, 1] = mol
        Chem.Kekulize(mol)
        bank[i, 0] = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)

    return bank


def prepare_seed(solutions, seed):
    """
    this method is depend on sorted score
    """

    solutions = solutions[np.where(solutions[:, 2] == True)]  # 사용 가능한 것이 True
    # x = np.argsort(5*solutions[:, 4]-solutions[:, 3])  # 5*QED - SAS
    x = np.argsort(obj_eq(solutions))  # coef.*QED - coef.*SAS
    solutions = x[::-1]
    # solutions[x[:nseed], 2] = False

    # shuffled_index = np.random.permutation(len(true_solutions))
    # true_solutions = true_solutions[shuffled_index]

    if len(solutions) > nseed:
        i = 0
        if len(seed) is 0:  # First selection,
            bank[solutions[0], 2] = False
            seed.append(bank[solutions[0]])
            i += 1

        if len(solutions) < len(seed):
            print(f"## Solutions is less than seeds / round {round_} / iter {niter}")
            raise ValueError

        while len(seed) < nseed:
            if len(solutions) == i + 1:
                print(
                    f"## Solutions is empty state / unused > nseed / # of seed: {len(seed)} / round {round_} / iter {niter}"
                )
                break
                # raise ValueError

            # dist = cal_array_dist(solutions[i, 1], seed)
            dist = np.zeros([len(seed)])
            for j in range(len(seed)):
                # fps1 = AllChem.GetMorganFingerprint(bank[solutions[i], 1], 2)
                fps1 = Chem.RDKFingerprint(bank[solutions[i], 1])
                # fps2 = AllChem.GetMorganFingerprint(seed[j][1], 2)
                fps2 = Chem.RDKFingerprint(seed[j][1])
                dist[j] = TanimotoSimilarity(fps1, fps2)
            if np.max(dist) > (1 - davg):
                i += 1
                # print(f'Dcut !!! {np.max(dist)}')
                continue
            else:
                bank[solutions[i], 2] = False
                seed.append(bank[solutions[i]])
                i += 1
    else:
        for i in bank[solutions]:
            seed.append(i)
        bank[solutions[:], 2] = False
        rnd_number = np.random.permutation(len(bank))
        i = 0
        while len(seed) <= nseed:
            if len(rnd_number) == i + 1:
                print(
                    f"## Solutions is empty state / unused < nseed / # of seed: {len(seed)} / round {round_} / iter {niter}"
                )
                break
            dist = np.zeros([len(seed)])
            for j in range(len(seed)):
                # fps1 = AllChem.GetMorganFingerprint(bank[rnd_number[i], 1], 2)
                fps1 = Chem.RDKFingerprint(bank[rnd_number[i], 1])
                # fps2 = AllChem.GetMorganFingerprint(seed[j][1], 2)
                fps2 = Chem.RDKFingerprint(seed[j][1])
                dist[j] = TanimotoSimilarity(fps1, fps2)
            if np.max(dist) > (1 - davg):
                i += 1
                # print(f'Dcut !!! {np.max(dist)}')
                continue
            else:
                seed.append(bank[rnd_number[i]])
                i += 1

    print(f"@ prepare_seed finished!")
    return np.asarray(seed)


def cut_smi(smi1, smi2, func, ring_bool):

    l_smi = None
    r_smi = None

    try:
        l_smi = func(smi1, "L", ring_bool, 4)
        r_smi = func(smi2, "R", ring_bool, 4)
    except (IndexError, ValueError):
        fail_f.write(f"{l_smi},{r_smi},piece\n")
        raise PermissionError

    return l_smi, r_smi


def crossover_smiles(smi1, smi2, func, ring_bool):
    new_smi = None
    mol = False
    l_smi = None
    r_smi = None

    try:
        l_smi, r_smi = cut_smi(smi1, smi2, func, ring_bool)

        k = 0
        while not mol:
            new_smi = ModSmi.tight_rm_branch(l_smi, r_smi)
            mol = Chem.MolFromSmiles(new_smi)
            k += 1
            if k == 30:
                break

        if not mol:
            l_smi, r_smi = cut_smi(smi2, smi1, func, ring_bool)

            k = 0
            while not mol:
                new_smi = ModSmi.tight_rm_branch(l_smi, r_smi)
                mol = Chem.MolFromSmiles(new_smi)
                k += 1
                if k == 30:
                    raise PermissionError

    except ValueError:
        # print(f'L_smi: {l_smi}, R_smi: {r_smi}')
        fail_f.write(f"{l_smi},{r_smi},np\n")

    return new_smi, mol


def append_seed(new_smi, mol, update_solution):
    try:
        new_solution = [new_smi, mol, True, calculateScore(mol), QED.default(mol)]
        update_solution.append(new_solution)
        return 1
    except Chem.rdchem.MolSanitizeException:
        print(f"#### QED error {new_smi}")
        return 0


def prepare_child(seed, nCross1=20, nCross2=20, nCross3=20, nReplace=20, nAdd=20, nRemove=20):
    update_solution = []
    # print(seed[:, 0])
    for i in range(seed.shape[0]):
        try:
            smi1 = seed[i, 0]
        except IndexError:
            print(f"seed_shape: {seed.shape} / #: {i}")
            raise Exception

        # CROSSOVER1 ###
        q = 0
        j = 0
        while j < nCross1:
            if q == 200:
                print(f"#### have problems in updating solutions @{smi1}")
                break
            try:
                w = np.random.randint(len(bank))
                smi2 = bank[w, 0]

                if np.random.random() >= 0.5:
                    new_smi, mol = crossover_smiles(smi1, smi2, ModSmi.prepare_rigid_crossover, True)
                else:
                    new_smi, mol = crossover_smiles(smi2, smi1, ModSmi.prepare_rigid_crossover, True)

                if mol:
                    j += append_seed(new_smi, mol, update_solution)
                    bank[w, 2] = False

            except PermissionError:
                q += 1

        # CROSSOVER2 ###
        q = 0
        j = 0
        while j < nCross2:
            if q == 200:
                print(f"#### have problems in updating solutions @{smi1}")
                break
            try:
                w = np.random.randint(len(bank))
                smi2 = bank[w, 0]

                if np.random.random() >= 0.5:
                    new_smi, mol = crossover_smiles(smi1, smi2, ModSmi.prepare_rigid_crossover, False)
                else:
                    new_smi, mol = crossover_smiles(smi2, smi1, ModSmi.prepare_rigid_crossover, False)

                if mol:
                    j += append_seed(new_smi, mol, update_solution)
                    bank[w, 2] = False

            except PermissionError:
                q += 1

        # # CROSSOVER3 ###
        # q = 0
        # j = 0
        # while j < nCross1:
        #     if q == 200:
        #         print(f"#### have problems in updating solutions @{smi1}")
        #         break
        #     try:
        #         w = np.random.randint(len(bank))
        #         smi2 = bank[w, 0]

        #         if np.random.random() >= 0.5:
        #             new_smi, mol = crossover_smiles(smi1, smi2, ModSmi.prepare_flexible_crossover)
        #         else:
        #             new_smi, mol = crossover_smiles(smi2, smi1, ModSmi.prepare_flexible_crossover)

        #         if mol:
        #             j += append_seed(new_smi, mol, update_solution)
        #             bank[w, 2] = False

        #     except PermissionError:
        #         q += 1

        # print(f"### seed_len1: {len(update_solution)}")

        # REPLACE ###
        q = 0
        j = 0
        while j < nReplace:
            if q == 200:
                print(f"#### have problems in updating solutions @{smi1}")
                break
            try:
                new_smi, mol = ModSmi.replace_atom(smi1)

                if mol:
                    j += append_seed(new_smi, mol, update_solution)

            except (PermissionError, Chem.rdchem.KekulizeException):
                q += 1

        # print(f"### seed_len2: {len(update_solution)}")

        # ADD ###
        q = 0
        j = 0
        while j < nAdd:
            if q == 200:
                print(f"#### have problems in updating solutions @{smi1}")
                break
            try:
                new_smi, mol = ModSmi.add_atom(smi1)

                if mol:
                    j += append_seed(new_smi, mol, update_solution)

            except PermissionError:
                q += 1

        # print(f"### seed_len3: {len(update_solution)}")

        # REMOVE ###
        q = 0
        j = 0
        while j < nRemove:
            if q == 200:
                print(f"#### have problems in updating solutions @{smi1}")
                break
            try:
                new_smi, mol = ModSmi.delete_atom(smi1)

                if mol:
                    j += append_seed(new_smi, mol, update_solution)
            except PermissionError:
                q += 1

    return np.asarray(update_solution)


def prepare_local_child(_smi, nReplace=10, nAdd=10, nRemove=10):
    _mol = Chem.MolFromSmiles(_smi)
    update_solution = [[_smi, _mol, True, calculateScore(_mol), QED.default(_mol)]]

    # REPLACE ###
    q = 0
    j = 0
    while j < nReplace:
        if q == 200:
            print(f"#### have problems in updating solutions @{_smi}")
            break
        try:
            new_smi, mol = ModSmi.replace_atom(_smi)
            if mol:
                j += append_seed(new_smi, mol, update_solution)
        except (PermissionError, Chem.rdchem.KekulizeException):
            q += 1

    # print(f"### seed_len2: {len(update_solution)}")

    # ADD ###
    q = 0
    j = 0
    while j < nAdd:
        if q == 200:
            print(f"#### have problems in updating solutions @{_smi}")
            break
        try:
            new_smi, mol = ModSmi.add_atom(_smi)
            if mol:
                j += append_seed(new_smi, mol, update_solution)
        except PermissionError:
            q += 1

    # print(f"### seed_len3: {len(update_solution)}")

    # REMOVE ###
    q = 0
    j = 0
    while j < nRemove:
        if q == 200:
            print(f"#### have problems in updating solutions @{_smi}")
            break
        try:
            new_smi, mol = ModSmi.delete_atom(_smi)
            if mol:
                j += append_seed(new_smi, mol, update_solution)
        except PermissionError:
            q += 1

    return np.asarray(update_solution)



def update_bank(child_solutions, local_opt=False):
    cnt_replace = 0
    # bank_min = np.min(5 * bank[:, 4] - bank[:, 3])  # 5*QED - SAS
    bank_min = np.min(obj_eq(bank))  # coef.*QED - coef.*SAS
    # child_solutions = child_solutions[(5 * child_solutions[:, 4] - child_solutions[:, 3]) > bank_min]  # 5*QED - SAS
    child_solutions = child_solutions[obj_eq(child_solutions) > bank_min]  # coef.*QED - coef.*SAS

    if len(child_solutions) == 0:
        raise PermissionError("child solutions 가 없습니다 !")
        # print(f'')

    for i in range(len(child_solutions)):
        if local_opt:
            local_solutions = prepare_local_child(child_solutions[i, 0])
            x = np.argmax(obj_eq(local_solutions))  # coef.*QED - coef.*SAS
            # fps1 = AllChem.GetMorganFingerprint(local_solutions[x, 1], 2)
            fps1 = Chem.RDKFingerprint(local_solutions[x, 1])
        else:
            # fps1 = AllChem.GetMorganFingerprint(child_solutions[i, 1], 2)
            fps1 = Chem.RDKFingerprint(child_solutions[i, 1])

        max_similarity = 0
        max_n = None
        for _ in range(len(bank)):
            # fps2 = AllChem.GetMorganFingerprint(bank[_, 1], 2)
            fps2 = Chem.RDKFingerprint(bank[_, 1])
            dist = TanimotoSimilarity(fps1, fps2)
            if dist > max_similarity:
                max_similarity = dist
                max_n = _
        
        if local_opt:
            if (1 - max_similarity) < dcut:
                if obj_eq(local_solutions[x : x + 1]) > obj_eq(bank[max_n : max_n + 1]):  # similarity check 없이 넣으면 같은 분자가 너무 많이 생성된다.
                    bank[max_n] = local_solutions[x : x + 1]
                    cnt_replace += 1
            else:
                _min = np.argmin(obj_eq(bank))
                # print("#"*10, obj_eq(local_solutions[x : x + 1]), obj_eq(bank[_min : _min + 1]), final_avg.mean())
                if (max_similarity < 0.98) and (obj_eq(bank[_min : _min + 1]) < final_avg.mean()):  # @3 추가 local6 에서 추가
                    # print("@"*10)
                    if obj_eq(local_solutions[x : x + 1]) > obj_eq(bank[_min : _min + 1]):
                        bank[_min] = local_solutions[x : x + 1]
                        cnt_replace += 1
        else:
            if (1 - max_similarity) < dcut:
                if obj_eq(child_solutions[i : i + 1]) > obj_eq(bank[max_n : max_n + 1]):
                    bank[max_n] = child_solutions[i]
                    cnt_replace += 1
            else:
                _min = np.argmin(obj_eq(bank))
                if (max_similarity < 0.98) and (obj_eq(bank[_min : _min + 1]) < final_avg.mean()):  # @3 추가 local6 에서 추가
                    if obj_eq(child_solutions[i : i + 1]) > obj_eq(bank[_min : _min + 1]):
                        bank[_min] = child_solutions[i]
                        cnt_replace += 1

    return cnt_replace


if __name__ == "__main__":

    target_value = args.value
    target_round = args.nconvergent

    R_d = 10 ** (np.log10(2 / target_value) / int(target_round))

    # @@
    nbank = args.nbank  # number of bank conformations
    # max_nbank = 500  # number of max bank conformations
    # min_nbank = 10  # number of min bank conformations
    # nbank_add = 0  # number of additional conformations collected at the beginning
    nseed = nbank / 2  # number of seeds(mating) per iteration
    # increase_bank = 0  # 0 = no increase  1 = increase by egap
    # increase_bank_energy_cut = 300.0
    # ecut_reduce = "fix"
    # max_increase = 9999
    # increase_bank_dcut = True  # if True the new conf should be farther than dcut from any conf in bank
    # max_repeat = 999999
    max_repeat = 300

    # a = ChkTime('load file')

    chk_load = ChkTime()

    # bank = init_bank('/home2/yongbeom/research/my_csa/all_pubchem_prob_0.01.smi.SAS', nbank, rseed=args.rseed)  # server PubChem

    bank = init_bank("/home2/yongbeom/research/my_csa/sampled_ZINC_0.001.smi.SAS", nbank, rseed=args.rseed)  # server ZINC

    # indv_PC
    # bank = init_bank("/home/yongbeom/research/gen_smiles/SAS/sampled_ZINC_0.001.smi.SAS", nbank, rseed=args.rseed)

    # INPUT: smiles, SAS, QED
    # smiles, mol format, used, SAS, QED, 5QED - SAS

    chk_load = chk_load.get()

    first_bank = bank

    origin_avg = obj_eq(bank)

    plot_list = []

    fail_f = open(f"fail_smiles.txt", "w")

    chk_calc = ChkTime()

    if nbank > 600:
        davg = cal_rnd_avg_dist(first_bank)
    else:
        davg = cal_avg_dist(first_bank)

    davg = 1 - davg
    davg = davg*args.distancecoef
    dcut = davg / 2

    final_avg = origin_avg

    chk_calc = chk_calc.get()

    with open(f"iteration.log", "w") as log_f2:
        log_f2.write(f"load_time: {chk_load:.3f} min\n")
        log_f2.write(f"dist_time: {chk_calc:.1f} min\n")
        log_f2.write(f"round  iter  unused  time_seed  time_child  time_update  n_replace\n")

    with open(f"message.log", "w") as log_f:
        log_f.write(f"nbank: {nbank}\n")
        log_f.write(f"nseed: {nseed}\n")
        log_f.write(f"max_repeat: {max_repeat}\n")
        log_f.write(f"R_d: {R_d:.6f} (convergent_round: {target_round})\n")
        log_f.write(f"D_avg: {davg:.3f} (similarity: {1-davg:.3f})\n")
        log_f.write(
            f"init_bank_avg - QED: {first_bank[:, 4].mean():.3f}, SAS: {first_bank[:, 3].mean():.3f}, OBJ: {origin_avg.mean():.3f}\n"
        )
        log_f.write(f"round   dcut  n_iter  obj_avg  obj_min  obj_max  n_replace  min/round\n")

    save_bank = np.empty([max_repeat, bank.shape[0], bank.shape[1] - 1], dtype=object)

    for round_ in range(max_repeat):
        if (round_ != 0) and (dcut > davg / 5):
            dcut *= R_d

        timechk = time.time()

        # print(f'dcut: {dcut}, davg: {davg}')
        niter = 0
        n_replace = 0
        iter_gate = True
        while iter_gate:
            seed = []
            # log_f.write(f'## SEED #### @ {np.count_nonzero(bank[:, 2] == True)} #############\n')
            time_seed = ChkTime()
            seed = prepare_seed(bank, seed)
            time_seed = time_seed.get()
            # log_f.write(f'## SEED #### @ {np.count_nonzero(bank[:, 2] == True)} #### AFTER ##\n')

            # log_f.write(f'## CHILD ### @ {np.count_nonzero(bank[:, 2] == True)} #############\n')
            time_child = ChkTime()
            child_solutions = prepare_child(seed)
            shuffled_index_ = np.random.permutation(child_solutions.shape[0])  # @4 에서 추가 됨.
            child_solutions = child_solutions[shuffled_index_]
            time_child = time_child.get()
            # log_f.write(f'## CHILD ### @ {np.count_nonzero(bank[:, 2] == True)} #### AFTER ##\n')

            time_update = ChkTime()
            try:
                # log_f.write(f'## BANK #### @ {np.count_nonzero(bank[:, 2] == True)} #############\n')
                # n_replace += update_bank(child_solutions, True)  # local update
                n_replace += update_bank(child_solutions)  # non-local update
                # log_f.write(f'## BANK #### @ {np.count_nonzero(bank[:, 2] == True)} #### AFTER ##\n')
            except PermissionError:
                break
            time_update = time_update.get()
            niter += 1

            if np.count_nonzero(bank[:, 2] == True) < (nbank - nseed * 0.9):
                iter_gate = False
            with open(f"iteration.log", "a") as log_f2:
                # log_f2.write(f'### round: {round_:2}, iter: {niter} repeat, # of unused: '
                #              f'{np.count_nonzero(bank[:, 2] == True)}, seed: {time_seed:.1f}, child: {time_child:.1f},'
                #              f' update: {time_update:.1f}')
                log_f2.write(
                    f"{round_:>4}  {niter:4}  {np.count_nonzero(bank[:, 2] == True):>6}     {time_seed:6.1f}"
                    f"      {time_child:6.1f}       {time_update:6.1f}    {n_replace:7}\n"
                )

        # final_avg = 5*bank[:, 4]-bank[:, 3]
        final_avg = obj_eq(bank)

        with open(f"message.log", "a") as log_f:
            # log_f.write(f'round: {round_:3>}, dcut: {dcut:.3f}, Obj avg.: {final_avg.mean():.3f}, '
            #             f'min.: {final_avg.min():.3f}, max.: {final_avg.max():.3f}, '
            #             f'{(time.time() - timechk)/60:.1f} min/round\n')
            # log_f.write(f'round  dcut  n_iter  obj_avg   obj_min   obj_max  n_replace  min/round\n')
            log_f.write(
                f"{round_:>4}   {dcut:4.3f}    {niter:3}   {final_avg.mean():6.3f}   {final_avg.min():6.3f}   "
                f"{final_avg.max():6.3f}    {n_replace:7}   {(time.time() - timechk)/60:8.2f}\n"
            )

        bank[:, 2] = True  # reset to unused solutions

        plot_list.append(final_avg.mean())
        tmp_bank = np.empty([bank.shape[0], bank.shape[1] - 1], dtype=object)
        tmp_bank[:, :3] = bank[:, [0, 3, 4]]
        tmp_bank[:, 3] = final_avg
        tmp_bank[:, 1:] = tmp_bank[:, 1:].astype(np.float16)
        tmp = np.argsort(tmp_bank[:, 3])  # 5*QED - SAS
        save_bank[round_] = tmp_bank[tmp[::-1]]
        
        df = pd.DataFrame(tmp_bank, columns=["SMILES", "SAS", "QED", "TARGET"])
        df.to_csv(f"bank_round{round_}_{R_d:.5f}_{args.coef:.3f}.csv", index=False)

    final_bank = np.empty([bank.shape[0], bank.shape[1] - 1], dtype=object)
    final_bank[:, :3] = bank[:, [0, 3, 4]]
    final_bank[:, 3] = final_avg
    final_bank[:, 1:] = final_bank[:, 1:].astype(np.float16)

    np.save(f"list_bank_{R_d:.5f}_{args.coef:.3f}.npy", save_bank)
    save_smiles = pd.DataFrame(save_bank[:, :, 0])
    save_smiles.to_csv(f"list_smiles_{R_d:.5f}_{args.coef:.3f}.csv", header=False, index=False)

    df = pd.DataFrame(final_bank, columns=["SMILES", "SAS", "QED", "TARGET"])
    df.to_csv(f"final_bank_{R_d:.5f}_{args.coef:.3f}.csv", index=False)

    # log_f.close()
    fail_f.close()

    plt.plot(plot_list)
    plt.tight_layout()
    plt.savefig("target_plot.png")

    # """
    # seed가 잘 선택 됐는지 확인.
    # """
    # print(seed)
    # test = np.zeros([len(seed), len(seed)])
    # for i in range(len(seed)-1):
    #     for j in range(i, len(seed)):
    #         fps1 = FingerprintMols.FingerprintMol(seed[i, 1])
    #         fps2 = FingerprintMols.FingerprintMol(seed[j, 1])
    #         test[i, j] = DataStructs.FingerprintSimilarity(fps1, fps2)
    #
    # print(test)

    # """
    # 96 smi2mol 0.000225 sec
    # 96 mol2fp 0.000369 sec
    # 96 smi2fp 0.000455 sec
    # """
    # a = ChkTime('smi2mol')
    # test = Chem.MolFromSmiles(j)
    # print(sys.getsizeof(test))
    # a.get()
    #
    # a = ChkTime('mol2fp')
    # test = FingerprintMols.FingerprintMol(test)
    # print(sys.getsizeof(test))
    # a.get()
    #
    # a = ChkTime('smi2fp')
    # test = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(j))
    # print(sys.getsizeof(test))
    # a.get()
