from itertools import combinations, chain

def power_set(iterable):
    """
    Generate the power set (without the empty set) for any given iterator
    """
    pset = chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))
    return list(list(combo) for combo in pset if len(combo) > 0)

if __name__ == "__main__":
    print(power_set([1, 2, 3]))