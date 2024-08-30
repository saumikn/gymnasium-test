import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import sys
from itertools import product
import os


class Tree:
    def __init__(self, depth=10, mode="float", seed=0):
        self.depth = depth
        self.n = 2**depth - 1
        if mode == "float":
            self.arr = np.random.default_rng(seed).random(self.n)
        elif mode == "linear":
            self.arr = np.random.default_rng(seed).random(self.n)
            for level in range(depth):
                self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= depth + 1
        elif mode == "binary" or mode.startswith("streak-"):
            self.arr = np.random.default_rng(seed).integers(
                0, 1, endpoint=True, size=self.n, dtype=np.int32
            )
        else:
            raise ValueError(
                "Mode must be 'float', 'linear', 'binary', or start with 'streak-"
            )

        self.mode = mode

        # self.arr = np.arange(n)
        # if factor is None:
        #     factor = n // 4
        # rng = np.random.default_rng(seed)
        # for _ in range(factor):
        #     a, b = rng.choice(n, size=2)
        #     self.arr[a], self.arr[b] = self.arr[b], self.arr[a]

    def _l(self, i):
        return i * 2 + 1

    def _r(self, i):
        return self._l(i) + 1

    # def get_maxsum(self, i, skill):
    #     if i >= self.n or skill < 0:
    #         return 0
    #     lsum = self.get_maxsum(self._l(i), skill - 1)
    #     rsum = self.get_maxsum(self._r(i), skill - 1)
    #     return self.arr[i] + max(lsum, rsum)

    # def get_choice(self, i, skill):
    #     if self._l(i) >= self.n:
    #         return None
    #     lsum = self.get_maxsum(self._l(i), skill - 1)
    #     rsum = self.get_maxsum(self._r(i), skill - 1)
    #     return int(rsum > lsum)

    def get_upstream(self, i):
        path = [i]
        while i > 0:
            i = (i - 1) // 2
            path.insert(0, i)
        return [p for p in path if p < self.n]

    def get_all_paths(self, i, skill):
        if i >= self.n or skill < 0:
            return None

        l = r = i
        for _ in range(skill):
            if self._l(l) < self.n:
                l = self._l(l)
                r = self._r(r)

        paths = [self.get_upstream(node) for node in range(l, r + 1)]
        paths = {p[-1]: p for p in paths}
        paths = list(paths.values())
        return paths

    # def crystal_paths(self, i, skill):
    #     return self._crystal_paths_helper(i, skill, 0, [])

    # def _crystal_paths_helper(self, i, skill, depth, paths):
    #     if depth > skill:
    #         return paths

    #     new_paths = []
    #     for path in paths:
    #         new_paths.append(path.copy() + [i])
    #     left_paths = self._crystal_paths_helper(self._l(i), skill, depth + 1, new_paths)
    #     right_paths = self._crystal_paths_helper(
    #         self._r(i), skill, depth + 1, new_paths
    #     )
    #     return left_paths.extend(right_paths)

    def score_path(self, path):
        scores = [self.arr[p] for p in path]
        if self.mode in ["float", "linear", "binary"]:
            return sum(scores)
        elif self.mode.startswith("streak-"):
            streak = int(self.mode.split("-")[1])
            for s, _ in enumerate(scores[: (1 - streak)]):
                if scores[s : s + streak] == [0] * streak:
                    scores[s : s + streak] = [2 for _ in range(streak)]
            return sum(scores)
        else:
            raise ValueError

    def get_scores(self, i, skill):
        if i >= self.n or skill < 0:
            return [0]
        paths = self.get_all_paths(i, skill)
        return [self.score_path(path) for path in paths]

    def get_choice(self, i, skill):
        if self._l(i) >= self.n or skill < 0:
            return None
        lscore = max(self.get_scores(self._l(i), skill - 1))
        rscore = max(self.get_scores(self._r(i), skill - 1))
        return int(rscore > lscore)

    def get_path(self, i, skill):
        path = []
        while i < self.n:
            choice = self.get_choice(i, skill)
            path += [(i, choice)]
            if choice is None:
                break
            i = self._l(i) + choice
        return path

    def get_training_data(self, skill):
        path = self.get_path(0, skill)[:-1]
        X = np.zeros((len(path), self.n * 2))
        X[:, : self.n] = self.arr.reshape((1, 1, -1))
        for i, (node, _) in enumerate(path):
            X[i, self.n + node] = 1
        Y = np.array([choice for (_, choice) in path])
        return X, Y

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return repr(self.arr)


class Forest:
    def __init__(self, depth, mode, start, stop):
        self.depth = depth
        self.m = stop - start
        self.trees = [Tree(depth, mode, seed) for seed in range(start, stop)]

    def get_training_data(self, skill):
        X, Y = [], []
        for tree in self.trees:
            _x, _y = tree.get_training_data(skill)
            X.append(_x)
            Y.append(_y)
        return np.concatenate(X), np.concatenate(Y)

    def eval_model(self, model):
        n = self.trees[0].n
        X = np.zeros((self.m, n * 2))
        for i, tree in enumerate(self.trees):
            X[i, :n] = tree.arr

        locs = np.zeros(self.m, dtype=np.int32)
        X[np.arange(self.m), locs + n] = 1

        paths = [locs]
        for i in range(self.depth - 1):
            Y = model(X).numpy().argmax(axis=1)
            X[np.arange(self.m), locs + n] = 0
            locs = locs * 2 + 1 + Y
            X[np.arange(self.m), locs + n] = 1
            paths.append(locs)
        paths = np.array(paths).T

        scores = []
        for tree, path in zip(self.trees, paths):
            scores.append(tree.score_path(path))
        return paths, scores, np.mean(scores) / self.m


class EarlyStopper:
    def __init__(self, patience=None, minimize=True):
        self.patience = patience
        self.wait = 0
        self.best_value = None
        self.comparator = lambda value, best_value: (
            value < best_value if minimize else value > best_value
        )

    def should_stop(self, value):
        at_best, at_patience = False, False
        if self.best_value is None or self.comparator(value, self.best_value):
            self.best_value = value
            self.wait = 0
            at_best = True
        else:
            self.wait += 1
        at_patience = self.patience and self.wait >= self.patience
        return at_best, at_patience


def make_model(depth, num_nodes=32, num_dense=3):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(2**depth - 1) * 2)
    x = tf.keras.layers.Flatten()(inputs)
    for _ in range(num_dense):
        x = tf.keras.layers.Dense(num_nodes, activation="relu")(x)
    output1 = tf.keras.layers.Dense(2, name="Y")(x)
    output1 = tf.keras.layers.Softmax()(output1)
    model = tf.keras.models.Model(inputs=inputs, outputs=output1)

    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"], run_eagerly=True)
    return model


def exp(seed, depth, mode, budget, skill, patience=10, write=True):
    X, Y = Forest(
        depth,
        mode,
        budget * seed,
        budget * (seed + 1),
    ).get_training_data(skill)

    test_budget = 1000
    test_offset = int(1e9)
    testing = Forest(
        depth,
        mode,
        test_budget * seed + test_offset,
        test_budget * (seed + 1) + test_offset,
    )
    model = make_model(depth)

    stopper = EarlyStopper(patience, minimize=False)
    for epoch in tqdm(range(50), disable=True):
        if epoch != 0:
            model.fit(X, Y, verbose=0)
        _, _, score = testing.eval_model(model)
        _, at_patience = stopper.should_stop(score)
        if write:
            write_output(
                seed,
                depth,
                mode,
                budget,
                skill,
                epoch,
                score,
                stopper.best_value,
                stopper.wait,
                stopper.patience,
            )
            with open("output_strat.txt", "a") as f:
                print(
                    f"{seed},{depth},{mode},{budget},{skill},{epoch},{score},{stopper.best_value},{stopper.wait},{stopper.patience}",
                    file=f,
                    flush=True,
                )
        if at_patience:
            break
    return score


def multi_exp(seeds, depths, modes, budgets, skills, pos=0):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["MPLCONFIGDIR"] = "/tmp/"
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    tqdm_kwargs = {
        "ncols": 80,
        "position": pos,
        "desc": f"Seeds {min(seeds)}-{max(seeds)}; Depth {depths[0]}",
        "leave": False,
    }
    combos = list(product(seeds, depths, modes, budgets, skills))
    for seed, depth, mode, budget, skill in tqdm(combos, **tqdm_kwargs):
        exp(seed, depth, mode, budget, skill, write=True)


def write_output(*args):
    with open("output_strat.txt", "a") as f:
        print(",".join([str(a) for a in args]), file=f, flush=True)


if __name__ == "__main__":
    WORKERS = 10

    depth = int(sys.argv[1])
    modes = sys.argv[2].split(",")
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    # budgets = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # budgets = [1, 10, 100, 1000, 10000]
    budgets = [1, 10, 100, 1000]
    # budgets = [10000]
    skills = [s for s in [2, 4, 6, 8, 10, 12] if s <= depth]
    # skills = [s for s in [12] if s <= depth]

    seeds = np.arange(start, end)
    seeds = np.split(seeds, WORKERS)

    combos = [
        (seed, [depth], modes, budgets, skills, i + 1) for i, seed in enumerate(seeds)
    ]

    process_kwargs = {
        "ncols": 80,
        "max_workers": WORKERS,
        "disable": False,
        "position": 0,
    }
    scores = process_map(multi_exp, *zip(*combos), **process_kwargs)
