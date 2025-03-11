from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import Atom, Mol
from typing import List, Dict, Optional

__all__ = ['get_atom_features', 'NUM_ATOM_FEATURES']

# 定义无机物的原子符号
METAL_SYMBOLS = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Fe', 'Cu', 'Zn'}
ATOM_SYMBOL = ('*', 'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Fe', 'Cu', 'Zn')
DEGREE = (0, 1, 2, 3, 4, 5, 6)  # 原子的度数
VALENCE = (0, 1, 2, 3, 4, 5, 6)  # 原子的价
FORMAL_CHARGE = (-3, -2, -1, 0, 1, 2, 3)  # 形式电荷
NUM_HS = (0, 1, 2, 3, 4)  # 连接的氢原子数
OXIDATION_STATE = (-3, -2, -1, 0, 1, 2, 3)  # 氧化态

# 电负性字典
EN = {
    '*': 0.00,
    'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.59, 'Cl': 3.16,
    'K': 0.82, 'Ca': 1.00, 'Fe': 1.83, 'Cu': 1.90, 'Zn': 1.65,
}

# 定义原子特征的信息
FEATURE_INFORM = OrderedDict([
    ['symbol', {'choices': ATOM_SYMBOL, 'allow_unknown': False}],  # 原子符号
    ['degree', {'choices': DEGREE, 'allow_unknown': True}],  # 度数
    ['valence', {'choices': VALENCE, 'allow_unknown': True}],  # 价
    ['formal_charge', {'choices': FORMAL_CHARGE, 'allow_unknown': True}],  # 形式电荷
    ['num_Hs', {'choices': NUM_HS, 'allow_unknown': True}],  # 连接的氢原子数
    ['oxidation_state', {'choices': OXIDATION_STATE, 'allow_unknown': True}],  # 氧化态
    ['mass', {'choices': None}],  # 原子质量
    ['EN', {'choices': None}],  # 电负性
])

# 计算每个特征的维度
for key, val in FEATURE_INFORM.items():
    if val['choices'] is None:
        val['dim'] = 1
    else:
        val['choices'] = {v: i for i, v in enumerate(val['choices'])}
        if val['allow_unknown']:
            val['dim'] = len(val['choices']) + 1
        else:
            val['dim'] = len(val['choices'])

# 计算总特征数
NUM_KEYS = len(FEATURE_INFORM)
NUM_ATOM_FEATURES = sum([val['dim'] for val in FEATURE_INFORM.values()])

def get_atom_features(atom: Atom, oxidation_state: Optional[int] = None) -> List[int]:
    """
    获取无机物原子的特征向量。
    :param atom: RDKit 的 Atom 对象
    :param oxidation_state: 原子的氧化态（如果未提供，则默认为形式电荷）
    :return: 原子的特征向量
    """

    symbol = atom.GetSymbol()
    is_metal = symbol in METAL_SYMBOLS  # 判断是否为金属原子

    # 处理金属原子的特殊特征
    if is_metal:
        degree = coordination_number if coordination_number is not None else atom.GetTotalDegree()  # 使用配位数或度数
        oxidation_state = oxidation_state if oxidation_state is not None else atom.GetFormalCharge()  # 使用氧化态或形式电荷
    else:
        degree = atom.GetTotalDegree()  # 非金属原子使用度数
        oxidation_state = oxidation_state if oxidation_state is not None else atom.GetFormalCharge()  # 使用氧化态或形式电荷

    symbol = atom.GetSymbol()
    features = {
        'symbol': symbol,
        'degree': atom.GetTotalDegree(),  # 原子的度数
        'valence': atom.GetTotalValence(),  # 原子的价
        'formal_charge': atom.GetFormalCharge(),  # 形式电荷
        'num_Hs': atom.GetTotalNumHs(),  # 连接的氢原子数
        'oxidation_state': oxidation_state if oxidation_state is not None else atom.GetFormalCharge(),  # 氧化态
        'mass': atom.GetMass() * 0.01,  # 原子质量（缩放）
        'EN': EN.get(symbol, 0.0) * 0.25,  # 电负性（缩放，默认值为 0.0）
    }
    return _get_sparse(features)

def _get_sparse(features: Dict) -> List[int]:
    """
    将原子特征转换为稀疏的 one-hot 编码向量。
    :param features: 原子特征字典
    :return: 稀疏的 one-hot 编码向量
    """
    retval = [0] * NUM_ATOM_FEATURES
    idx = 0
    for key, inform in FEATURE_INFORM.items():
        choices, dim = inform['choices'], inform['dim']
        x = features[key]
        if choices is None:
            retval[idx] = x
        elif inform['allow_unknown'] is True:
            retval[idx + choices.get(x, dim - 1)] = 1
        else:
            retval[idx + choices[x]] = 1
        idx += dim
    return retval