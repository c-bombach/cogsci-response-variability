
import numpy as np
from abc import ABC,abstractmethod
from numpy import linalg as LA
import ccobra
import pandas as pd
from typing import List, Dict, Set

### PREMISE/RESPONSE ENCODINGS and METRICS ###

def generate_all_syllogisms() -> List[str]:
    """Generate all 64 syllogisms programmatically"""
    quantifiers = ['A', 'E', 'I', 'O']
    figures = ['1', '2', '3', '4']
    
    syllogisms = []
    for q1 in quantifiers:
        for q2 in quantifiers:
            for fig in figures:
                syllogisms.append(f"{q1}{q2}{fig}")
    return syllogisms

def generate_all_responses() -> List[str]:
    """Generate all 9 responses"""
    quantifiers = ['A', 'E', 'I', 'O']
    directions = ['ac','ca']
    responses = ['NVC']
    for q in quantifiers:
        for d in directions:
            responses.append(f"{q}{d}")
    return responses

ALL_SYLLOGISMS = generate_all_syllogisms()
ALL_RESPONSES = generate_all_responses()

# Get correct responses from CCOBRA
FOL_REASONER = ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES
CORRECT_RESPONSES = {key: set(value) for key, value in FOL_REASONER.items()}

INVALID_SYLLOGISMS_SET = {s for s in ALL_SYLLOGISMS if CORRECT_RESPONSES[s] == {'NVC'}}
VALID_SYLLOGISMS_SET = set(ALL_SYLLOGISMS) - INVALID_SYLLOGISMS_SET

### Properties encoding ###

class TaskEncoder(ABC):
    @abstractmethod
    def encode(self, syllog: str) -> np.ndarray:
        """Return a fixed-length vector representation."""
        pass

    def distance(self, syllog1: str, syllog2: str) -> float:
        """Default distance: L1 norm on encoded vectors."""
        return LA.norm(self.encode(syllog1) - self.encode(syllog2), 1)

    def norm(self, syllog: str) -> float:
        """Default norm: L1 norm of the encoded vector."""
        return LA.norm(self.encode(syllog), 1)


class ResponseEncoder(ABC):
    @abstractmethod
    def encode(self, resp: str) -> np.ndarray:
        pass

    def distance(self, resp1: str, resp2: str) -> float:
        return LA.norm(self.encode(resp1) - self.encode(resp2), 1)

    def norm(self, resp: str) -> float:
        return LA.norm(self.encode(resp), 1)

class StandardTaskEncoder(TaskEncoder):
    def __init__(self):
        self.quantifier_properties = {
            'A': {'universal': 1, 'affirmative': 1, 'reversible': 0},
            'E': {'universal': 1, 'affirmative': 0, 'reversible': 1},
            'I': {'universal': 0, 'affirmative': 1, 'reversible': 1},
            'O': {'universal': 0, 'affirmative': 0, 'reversible': 0}
        }
        
        self.figure_properties = {
            "1": {"transitive": 1, "forward": 1},
            "2": {"transitive": 1, "forward": 0},
            "3": {"transitive": 0, "forward": 1},
            "4": {"transitive": 0, "forward": 0}
        }
    def encode(self,syllog: str) -> np.ndarray:
        return np.array([
            self.quantifier_properties[syllog[0]]["universal"],
            self.quantifier_properties[syllog[0]]["affirmative"],
            self.quantifier_properties[syllog[1]]["universal"],
            self.quantifier_properties[syllog[1]]["affirmative"],
            self.figure_properties[syllog[2]]["transitive"],
            self.figure_properties[syllog[2]]["forward"]
        ])
        

class StandardResponseEncoder(ResponseEncoder):
    def __init__(self):
        self.quantifier_properties = {
            'A': {'universal': 1, 'affirmative': 1, 'reversible': 0},
            'E': {'universal': 1, 'affirmative': 0, 'reversible': 1},
            'I': {'universal': 0, 'affirmative': 1, 'reversible': 1},
            'O': {'universal': 0, 'affirmative': 0, 'reversible': 0}
        }
        self.direction_properties = {
            "ac": {"is_ac": 1, "is_ca": 0},
            "ca": {"is_ac": 0, "is_ca": 1}
        }
    def encode(self,resp: str) -> np.ndarray:
        if resp != "NVC":
            return np.array([
                self.quantifier_properties[resp[0]]["universal"],
                self.quantifier_properties[resp[0]]["affirmative"],
                self.direction_properties[resp[1:]]["is_ac"],
                self.direction_properties[resp[1:]]["is_ca"]
            ])
        else:
            return np.array([0, 0, 0, 0])

class OneHotTaskEncoder(TaskEncoder):
    def __init__(self, all_syllogisms):
        self.all_syllogisms = sorted(all_syllogisms)
        self.idx = {s: i for i, s in enumerate(self.all_syllogisms)}
        self.dim = len(self.all_syllogisms)

    def encode(self, syllog: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        vec[self.idx[syllog]] = 0.5
        return vec

class OneHotResponseEncoder(ResponseEncoder):
    def __init__(self, all_responses):
        self.all_responses = sorted(all_responses)
        self.idx = {s: i for i, s in enumerate(self.all_responses)}
        self.dim = len(self.all_responses)

    def encode(self, syllog: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        vec[self.idx[syllog]] = 0.5
        return vec

def compute_neighbors(syllogisms: List[str], encoder: TaskEncoder, r: float = 1) -> Dict[str, Set[str]]:
    """Precompute all tasks at distance r according to the given encoder (effectively the punctured r-ball)."""
    ball_dict = {}
    for s1 in syllogisms:
        neighbors = []
        for s2 in syllogisms:
            if s1 != s2:
                dist = encoder.distance(s1,s2)
                if dist <= r:
                    neighbors.append(s2)
        ball_dict[s1] = set(neighbors)
    return ball_dict

def local_energy(pattern: Dict[str,str],
                 task: str,
                 task_encoder: TaskEncoder,
                 resp_encoder: ResponseEncoder,
                 p: float = 2) -> float:
    "Computes local energy of the given response pattern at the given task"
    
    tasks = list(pattern.keys())
    neighbors = compute_neighbors(tasks,task_encoder)
    dists = []
    for nb in neighbors[task]:
        d = resp_encoder.distance(pattern[task],pattern[nb])
        dists.append(d ** p if p != float('inf') else d)
    if not dists:
        return 0
    if p == float('inf'):
        return max(dists)
    if p == float('inf'):
        return max(dists)
    else:
        return sum(dists) / (len(dists))

def global_energy(pattern: Dict[str,str],
                  task_encoder: TaskEncoder,
                  resp_encoder: ResponseEncoder,
                  p: float = 2) -> float:
    total_energy = 0.0
    for task in pattern:
            total_energy += local_energy(pattern,task,task_encoder,resp_encoder,p)
    return total_energy

def lp_distance(pattern1: Dict[str,str],
                pattern2: Dict[str,str],
                resp_encoder: ResponseEncoder,
                p: float = 2) -> float:
    lp_dist = 0.0
    for task in pattern1 and pattern2:
        lp_dist += resp_encoder.distance(pattern1[task], pattern2[task]) ** p
    return lp_dist ** (1/p)


def compute_reasoner_dict(reasoner_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Convert DataFrame to response dictionary
    Assumes 'enc_response' column contains sets
    """
    # Directly convert to dict - enc_response should already be sets
    return dict(zip(reasoner_df["enc_task"], reasoner_df["enc_response"]))



