from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 1e-3
    niter: int = 2000
    print_every: int = 100

@dataclass
class GeoConfig:
    k: int = 10              # kNN for graph on E
    epsilon: float = 1e-9    # jitter for SPD
    use_symmetric_edge: bool = True

@dataclass
class GWConfig:
    # No parameters needed here for vanilla POT's GW, left for extension
    pass