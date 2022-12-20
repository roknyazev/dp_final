import dataclasses
import random

random.seed(42)

capacity_small = 25
capacity_medium = 100
capacity_large = 400

km_price_small = 1
km_price_medium = 5
km_price_large = 12


@dataclasses.dataclass
class Info:
    capacity: float
    km_price: float


type_info = {0: Info(capacity_small, km_price_small),
             1: Info(capacity_medium, km_price_medium),
             2: Info(capacity_large, km_price_large)}
