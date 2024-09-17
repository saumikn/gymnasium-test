import numpy as np
from tqdm import tqdm
import sys
from itertools import product
import os
import gc
import time

# from memory_profiler import profile
from pympler import tracker


class Tree:
    def __init__(self, depth=10, mode="float", seed=0):
        self.depth = depth
        self.n = 2 ** (self.depth) - 1
        self.mode = mode
        self.get_all_path_cache = {}
        self.get_visible_arr_cache = {}
        if mode == "float":
            self.arr = np.random.default_rng(seed).random(self.n)
        elif mode.startswith("linear-"):
            factor = float(mode.split("-")[1])
            self.arr = np.random.default_rng(seed).random(self.n)
            for level in range(self.depth):
                self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= level * factor
        elif mode.startswith("exponential-"):
            factor = float(mode.split("-")[1])
            self.arr = np.random.default_rng(seed).random(self.n)
            for level in range(self.depth):
                self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= factor**level
        # elif mode == "linear":
        #     self.arr = np.random.default_rng(seed).random(self.n)
        #     for level in range(depth):
        #         self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= depth + 1
        elif mode == "binary" or mode.startswith("streak-"):
            self.arr = np.random.default_rng(seed).integers(
                0, 1, endpoint=True, size=self.n, dtype=np.int32
            )
        elif mode.startswith("risk-"):
            self.arr = np.zeros(self.n, dtype=np.float32)
            self.arr[self.n // 2] = float(mode[5:])
            self.arr[np.random.default_rng(seed).integers(self.n // 2, self.n - 1)] = 1
        else:
            raise ValueError(
                "Mode must be 'float', 'linear', 'binary', or start with 'streak-"
            )

    @staticmethod
    def _l(i):
        return i * 2 + 1

    @staticmethod
    def _r(i):
        return Tree._l(i) + 1

    def get_upstream(self, i):
        path = [i]
        while i > 0:
            i = (i - 1) // 2
            path.insert(0, i)
        return [p for p in path if p < self.n]

    def get_all_paths(self, i, skill):
        if (i, skill) in self.get_all_path_cache:
            return self.get_all_path_cache[(i, skill)]
        if i >= self.n or skill < 0:
            self.get_all_path_cache[(i, skill)] = None
            return None

        l = r = i
        for _ in range(skill):
            if Tree._l(l) < self.n:
                l = Tree._l(l)
                r = Tree._r(r)

        paths = [self.get_upstream(node) for node in range(l, r + 1)]
        paths = {p[-1]: p for p in paths}
        paths = list(paths.values())
        self.get_all_path_cache[(i, skill)] = paths
        return paths

    def score_path(self, path):
        scores = [self.arr[p] for p in path]
        if (
            self.mode in ["float", "binary"]
            or self.mode.startswith("linear-")
            or self.mode.startswith("exponential-")
            or self.mode.startswith("risk-")
        ):
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
        if Tree._l(i) >= self.n or skill < 0:
            return None
        lscore = max(self.get_scores(Tree._l(i), skill - 1))
        rscore = max(self.get_scores(Tree._r(i), skill - 1))
        return int(rscore > lscore)

    def get_path(self, i, skill):
        path = []
        while i < self.n:
            choice = self.get_choice(i, skill)
            path += [(i, choice)]
            if choice is None:
                break
            i = Tree._l(i) + choice
        return path

    def get_visible_arr(self, window, i):
        if (window, i) in self.get_visible_arr_cache:
            return self.get_visible_arr_cache[(window, i)]
        up = self.get_upstream(i)[:-1]
        split = self.depth - 2
        X = np.zeros(self.depth + 2 ** (window + 1) - 3) - 1
        X[split - len(up) : split] = [self.arr[u] for u in up]
        paths = self.get_all_paths(i, window)
        visible = list(sorted(list(set(j for i in paths for j in i))))
        visible = [self.arr[v] for v in visible if v not in up]
        X[split : split + len(visible)] = visible
        self.get_visible_arr_cache[(window, i)] = X
        return X

    def get_training_data(self, skill, window=0):

        path = self.get_path(0, skill)[:-1]

        if window:
            X = np.stack([self.get_visible_arr(window, i) for (i, _) in path])
        else:
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

    # @profile
    def get_training_data(self, skill, window=0):
        X, Y = [], []
        for tree in self.trees:
            _x, _y = tree.get_training_data(skill, window)
            X.append(_x)
            Y.append(_y)
        X_array = np.concatenate(X)
        Y_array = np.concatenate(Y)
        del X
        del Y
        return X_array, Y_array

    def eval_model(self, model, window):

        if window == 0:
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

        else:
            paths = [[0] for _ in enumerate(self.trees)]
            for _ in range(self.depth - 1):
                X = np.stack(
                    [
                        tree.get_visible_arr(window, paths[i][-1])
                        for i, tree in enumerate(self.trees)
                    ]
                )
                Y = model(X).numpy().argmax(axis=1)
                for i, choice in enumerate(Y):
                    paths[i].append(Tree._l(paths[i][-1]) + choice)

        scores = []
        for tree, path in zip(self.trees, paths):
            scores.append(tree.score_path(path))
        return paths, scores, np.mean(scores)


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


def make_model(depth, window, opt="sgd", lr=1, num_nodes=16, num_layers=1):
    import tensorflow as tf

    if window == 0:
        inputs = tf.keras.layers.Input(shape=(2**depth - 1) * 2)
    else:
        inputs = tf.keras.layers.Input(shape=depth + 2 ** (window + 1) - 3)
    x = tf.keras.layers.Flatten()(inputs)
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(num_nodes, activation="relu")(x)
    output1 = tf.keras.layers.Dense(2, name="Y")(x)
    output1 = tf.keras.layers.Softmax()(output1)
    model = tf.keras.models.Model(inputs=inputs, outputs=output1)

    if opt == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=0.001 * lr)
    elif opt == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=0.01 * lr)
    elif opt == "sgdn":
        opt = tf.keras.optimizers.SGD(
            learning_rate=0.01 * lr, momentum=0.90, nesterov=True
        )
    else:
        raise ValueError("Optimizer should be either 'adam' or 'sgd'")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"], run_eagerly=False)
    return model


# @profile
def exp(
    seed,
    decisions,
    window,
    mode,
    budget,
    skill,
    opt,
    lr,
    patience=20,
    write=True,
    num_nodes=16,
    num_layers=1,
):
    depth = decisions + 1
    print(f"\n\n\nStarting experiment: Seed-{seed} Depth-{depth} Skill-{skill}")

    import tensorflow as tf

    st = time.perf_counter()
    train = Forest(depth, mode, budget * seed, budget * (seed + 1))
    X_train, Y_train = train.get_training_data(skill, window=window)
    X_train, Y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train)
    print(f"train data: {seed} {depth} {skill}", time.perf_counter() - st)
    # return

    st = time.perf_counter()
    test_budget, test_offset = 2000, int(1e9)
    test = Forest(
        depth,
        mode,
        test_budget * seed + test_offset,
        test_budget * (seed + 1) + test_offset,
    )
    print("made testing forest")
    X_test, Y_test = test.get_training_data(skill, window=window)
    print("made testing data")
    X_test, Y_test = tf.convert_to_tensor(X_test), tf.convert_to_tensor(Y_test)
    print("made testing tensor")
    print(f"test data: {seed} {depth} {skill}", time.perf_counter() - st)

    st = time.perf_counter()
    model = make_model(depth, window, opt, lr, num_nodes, num_layers)
    print(f"make model: {seed} {depth} {skill}", time.perf_counter() - st)

    tqdm_kwargs = {
        "ncols": 80,
        "leave": True,
        "desc": f"Seed {seed} Depth {depth} Skill {skill}",
        # "position": 1,
        # "disable": True,
    }
    stopper = EarlyStopper(patience, minimize=False)

    for epoch in tqdm(range(2000), **tqdm_kwargs):
        # print()
        # st = time.perf_counter()
        if epoch != 0:
            model.fit(X_train, Y_train, batch_size=256, verbose=0)
        # print(f"model fit: {seed} {epoch}", time.perf_counter() - st)

        # st = time.perf_counter()
        loss, acc = model.evaluate(X_test, Y_test, batch_size=4096, verbose=0)
        # print(f"model evaluate: {seed} {epoch}", time.perf_counter() - st)

        # st = time.perf_counter()
        _, scores, _ = test.eval_model(model, window=window)
        valid_score = np.mean(scores[: test_budget // 2])
        test_score = np.mean(scores[test_budget // 2 :])
        # print(f"model eval_model: {seed} {epoch}", time.perf_counter() - st)
        _, at_patience = stopper.should_stop(valid_score)

        if write:
            write_output(
                seed,
                depth - 1,
                window,
                mode,
                budget,
                skill,
                opt,
                lr,
                num_nodes,
                num_layers,
                epoch,
                loss,
                acc,
                valid_score,
                stopper.wait,
                stopper.patience,
                test_score,
            )
        if at_patience:
            break

    st = time.perf_counter()
    del X_train, Y_train, X_test, Y_test
    del train, test
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"deleting: {seed} {depth} {skill}", time.perf_counter() - st)


# @profile
def single_exp(seeds, decisions, windows, modes, budgets, skills, opts, lrs):
    print("importing tensorflow")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["MPLCONFIGDIR"] = "/tmp/"
    import tensorflow as tf
    import numpy as np

    tf.config.set_visible_devices([], "GPU")

    print("finished importing tensorflow")

    tqdm_kwargs = {"ncols": 80, "leave": True, "disable": True}

    combos = list(product(seeds, decisions, windows, modes, budgets, skills, opts, lrs))
    for combo in tqdm(combos, **tqdm_kwargs):
        exp(*combo)


def write_output(*args):
    with open("/storage1/fs1/chien-ju.ho/Active/tree/output15.txt", "a") as f:
        print(",".join([str(a) for a in args]), file=f, flush=True)


def main(decisions, window, skills, mode, budget, start, end, opt, lr):
    single_exp(
        np.arange(start, end),
        [decisions],
        [window],
        [mode],
        [budget],
        skills,
        [opt],
        [lr],
    )


if __name__ == "__main__":

    decisions = int(sys.argv[1])
    mode = sys.argv[2]
    budget = int(sys.argv[3])
    start, end = [int(i) for i in sys.argv[4].split("-")]
    opt = sys.argv[5]
    lr = float(sys.argv[6])

    # window = min(12, decisions)
    # skills = [s for s in [2, 4, 6, 8] if s <= window]
    window = 0
    num = 4
    assert decisions % num == 0, f"{decisions} and {num} aren't divisible"
    skills = decisions // num * np.arange(num + 1)

    main(decisions, window, skills, mode, budget, start, end, opt, lr)

    # budgets = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # budgets = [1, 10, 100, 1000, 10000]
    # budgets = [10000]
    # skills = [s for s in [2, 4, 6, 8, 10, 12] if s <= depth]
    # skills = [s for s in [2, 4, 6, 8] if s <= depth]
    # skills = [s for s in [12] if s <= depth]

    # seeds = np.arange(start, end)
    # seeds = np.split(seeds, WORKERS)

    # combos = [
    #     (seed, [depth], modes, budgets, skills, [opt], [lr], i + 1)
    #     for i, seed in enumerate(seeds)
    # ]

    # process_kwargs = {
    #     "ncols": 80,
    #     "max_workers": WORKERS,
    #     "disable": False,
    #     "position": 0,
    # }
    # scores = process_map(multi_exp, *zip(*combos), **process_kwargs)
