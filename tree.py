print("importing numpy")

import numpy as np
from tqdm import tqdm
import sys
from itertools import product
import os
import gc
import time

# from memory_profiler import profile
# from pympler import tracker

print("importing tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

print("done with imports")


def softmax(a, temp):
    a /= temp
    a -= a.max()
    a = np.exp(a) / np.exp(a).sum()
    return a


skill_key = {
    "A": (1, 0),
    "B": (2, 0),
    "C": (3, 0),
    "D": (4, 0),
}


class Tree:
    def __init__(self, depth=10, mode="float", seed=0):
        self.depth = depth
        self.n = 2 ** (self.depth) - 1
        self.mode = mode
        self.get_all_path_cache = {}
        self.get_visible_arr_cache = {}
        self.rng = np.random.default_rng(seed)
        if mode == "float":
            self.arr = self.rng.random(self.n)
        elif mode.startswith("linear-"):
            factor = float(mode.split("-")[1])
            self.arr = self.rng.random(self.n)
            for level in range(self.depth):
                self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= level * factor
        elif mode.startswith("exponential-"):
            factor = float(mode.split("-")[1])
            self.arr = self.rng.random(self.n)
            for level in range(self.depth):
                self.arr[2**level - 1 : 2 ** (level + 1) - 1] *= factor**level
        elif mode == "binary" or mode.startswith("streak-"):
            self.arr = self.rng.integers(
                0, 1, endpoint=True, size=self.n, dtype=np.int32
            )
        elif mode.startswith("risk-"):
            self.arr = np.zeros(self.n, dtype=np.float32)
            self.arr[self.n // 2] = float(mode[5:])
            self.arr[self.rng.integers(self.n // 2, self.n - 1)] = 1
        elif mode.startswith("normal-"):
            self.arr = np.zeros(self.n, dtype=np.float32)
            self.arr[0::2] = self.rng.normal(0, 1, self.arr[0::2].shape)
            loc = int(mode.split("-")[1])
            self.arr[1::2] = self.rng.normal(loc, 1, self.arr[1::2].shape)
            self.arr[self.rng.integers(self.n // 2, self.n - 1)] += (
                depth * int(mode.split("-")[2]) * 2
            )
        elif mode.startswith("uniform-"):
            self.arr = np.zeros(self.n, dtype=np.float32)
            self.arr[0::2] = self.rng.uniform(0, 1, self.arr[0::2].shape)
            a = int(mode.split("-")[1]) / 100
            b = int(mode.split("-")[2]) / 100
            self.arr[1::2] = self.rng.uniform(a, b, self.arr[1::2].shape)
        elif mode.startswith("crit-"):
            self.arr = self.rng.random(self.n)
            rewards = self.rng.random(self.n // 2)
            a, b, c, d = [float(i) for i in mode.split("-")[1:]]
            for i, reward in enumerate(rewards):
                l = self._l(i)
                r = self._r(i)
                if reward < d / 2:
                    self.arr[i] += -a
                    self.arr[l] += b
                    self.arr[r] += -c
                elif reward < d:
                    self.arr[i] += -a
                    self.arr[l] += -c
                    self.arr[r] += b

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

    def get_all_paths(self, i, vision):
        if (i, vision) in self.get_all_path_cache:
            return self.get_all_path_cache[(i, vision)]
        if i >= self.n or vision < 0:
            self.get_all_path_cache[(i, vision)] = None
            return None

        l = r = i
        for _ in range(vision):
            if Tree._l(l) < self.n:
                l = Tree._l(l)
                r = Tree._r(r)

        paths = [self.get_upstream(node) for node in range(l, r + 1)]
        paths = {p[-1]: p for p in paths}
        paths = list(paths.values())
        self.get_all_path_cache[(i, vision)] = paths
        return paths

    def score_path(self, path):
        scores = [self.arr[p] for p in path]
        if self.mode.startswith("streak-"):
            streak = int(self.mode.split("-")[1])
            for s, _ in enumerate(scores[: (1 - streak)]):
                if scores[s : s + streak] == [0] * streak:
                    scores[s : s + streak] = [2 for _ in range(streak)]
            return sum(scores)
        else:
            return sum(scores)

    def get_scores(self, i, vision):
        if i >= self.n or vision < 0:
            return [0]
        paths = self.get_all_paths(i, vision)
        return [self.score_path(path) for path in paths]

    def get_choice(self, i, skill):
        vision, temp = skill_key[skill]
        if Tree._l(i) >= self.n or vision < 0:
            return None
        lscore = max(self.get_scores(Tree._l(i), vision - 1))
        rscore = max(self.get_scores(Tree._r(i), vision - 1))

        scores = np.array([lscore, rscore])
        if temp > 0:
            scores = softmax(scores, temp)
            return self.rng.choice(2, p=scores)
        else:
            return np.argmax(scores)

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

    def get_training_data(self, skill, window=0, leaf=False):

        if leaf:
            path = self.get_path(0, skill)
        else:
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
    def get_training_data(self, skill, window=0, leaf=False):
        X, Y = [], []
        for tree in self.trees:
            _x, _y = tree.get_training_data(skill, window, leaf)
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


def make_model(
    depth=None, window=0, opt="sgd", lr=1, num_nodes=16, num_layers=1, model=None
):
    if model is None:
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
    skill_str,
    budget,
    patience=20,
    write=True,
    opt="sgd",
    lr=1,
    num_nodes=16,
    num_layers=1,
    verbose=True,
    perf_stopping=False,
):
    depth = decisions + 1
    if verbose:
        print(f"\n\n\nStarting experiment: Seed-{seed} Depth-{depth} Skill-{skill_str}")

    st = time.perf_counter()
    model = make_model(depth, window, opt, lr, num_nodes, num_layers)
    best = model.get_weights()

    if decisions == 4:
        if isinstance(budget, (list, np.ndarray)):
            budget = np.array(budget).astype(int)
            budget_str = "_".join(str(i) for i in budget)

        elif budget.startswith("right_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.1, 0.125, 0.25, 0.5])

        elif budget.startswith("equal_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.25, 0.25, 0.25, 0.25])

        elif budget.startswith("left_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.5, 0.25, 0.125, 0.125])

        budget = budget.astype(int)
        budget = dict(zip("ABCD", budget))
        if verbose:
            print(budget)

    elif decisions == 2:
        if isinstance(budget, list):
            budget = np.array(budget).atype(int)
            budget_str = "_".join(str(i) for i in budget)

        elif budget.startswith("right_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.25, 0.75])

        elif budget.startswith("equal_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.5, 0.5])

        elif budget.startswith("left_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.75, 0.25])

        budget = budget.astype(int)
        budget = dict(zip("AB", budget))
        if verbose:
            print(budget)

    if verbose:
        print(
            f"make model: {seed} {depth} {skill_str} {budget_str}",
            time.perf_counter() - st,
        )

    skills = [[(s, budget[s]) for s in stage] for stage in skill_str.split("_")]
    if verbose:
        print(skills)

    if perf_stopping:
        stopper = EarlyStopper(patience, minimize=False)
        best_test = -np.inf

    else:
        stopper = EarlyStopper(patience, minimize=True)
        best_test = np.inf

    for stagei, stage in enumerate(skills):
        if verbose:
            print(stage)
        model = make_model(model=model)
        model.set_weights(best)

        X_trains, Y_trains, X_tests, Y_tests = [], [], [], []

        train_offset = int(1e8)
        test_budget, test_offset = 16384, int(1e9)

        for si, (s, budget) in enumerate(stage):
            st = time.perf_counter()
            if not budget:
                continue
            train = Forest(
                depth,
                mode,
                budget * seed + si * train_offset,
                budget * (seed + 1) + si * train_offset,
            )
            X_train, Y_train = train.get_training_data(s, window=window)
            if verbose:
                print(
                    f"train data: {seed} {depth} {s} {budget}", time.perf_counter() - st
                )

            st = time.perf_counter()
            # test_budget, test_offset = 4096, int(1e9)
            test = Forest(
                depth,
                mode,
                test_budget * seed + si * train_offset + test_offset,
                test_budget * (seed + 1) + si * train_offset + test_offset,
            )
            X_test, Y_test = test.get_training_data(s, window=window)
            if verbose:
                print(
                    f"test data: {seed} {depth} {s} {budget}", time.perf_counter() - st
                )
            X_trains.append(X_train)
            Y_trains.append(Y_train)
            X_tests.append(X_test)
            Y_tests.append(Y_test)

        if not X_trains:
            continue

        X_train = tf.convert_to_tensor(np.concatenate(X_trains))
        Y_train = tf.convert_to_tensor(np.concatenate(Y_trains))
        X_test = tf.convert_to_tensor(np.concatenate(X_tests))
        Y_test = tf.convert_to_tensor(np.concatenate(Y_tests))

        tqdm_kwargs = {
            "ncols": 80,
            "leave": True,
            "desc": f"Seed {seed} Depth {depth} Stage {'+'.join(s for s, _ in stage)}",
            # "position": 1,
            "disable": not verbose,
        }

        stopper.wait = 0
        for epoch in tqdm(range(10000), **tqdm_kwargs):
            # print()
            # st = time.perf_counter()
            if epoch != 0:
                model.fit(X_train, Y_train, batch_size=64, verbose=0)
            # print(f"model fit: {seed} {epoch}", time.perf_counter() - st)

            # st = time.perf_counter()
            loss, acc = model.evaluate(X_test, Y_test, batch_size=16384, verbose=0)
            # print(f"model evaluate: {seed} {epoch}", time.perf_counter() - st)

            if perf_stopping:
                st = time.perf_counter()
                _, scores, _ = test.eval_model(model, window=window)
                valid_score = np.mean(scores[: test_budget // 2])
                test_score = np.mean(scores[test_budget // 2 :])
            else:
                valid_score, test_score = 0, 0
            # print(f"model eval_model: {seed} {epoch}", time.perf_counter() - st)
            at_best, at_patience = stopper.should_stop(loss)
            if at_best:
                best = model.get_weights()
                best_output = [
                    seed,
                    depth - 1,
                    window,
                    mode,
                    stagei,
                    skill_str,
                    budget_str,
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
                ]

            if at_patience:
                if perf_stopping == False:
                    model.set_weights(best)
                    _, scores, _ = test.eval_model(model, window=window)
                    best_test = np.mean(scores)
                    best_output[17] = best_test

                if write:
                    write_output(get_output(best_output))
                break

    del X_train, Y_train, X_test, Y_test
    del train, test
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    if verbose:
        print(
            f"deleting: {seed} {depth} {skill_str} {budget_str}",
            time.perf_counter() - st,
        )
    return stopper.best_value, best_test


def get_output(args):
    return ",".join([str(a) for a in args])


def write_output(output):
    with open("/storage1/fs1/chien-ju.ho/Active/tree/output28.txt", "a") as f:
        print(output, file=f, flush=True)


# @profile
def single_exp(seeds, decisions, windows, modes, skills, budgets):

    tqdm_kwargs = {"ncols": 80, "leave": True, "disable": True}

    combos = list(product(seeds, decisions, windows, modes, skills, budgets))
    print(len(combos))
    for combo in tqdm(combos, **tqdm_kwargs):
        exp(*combo)


def main(decisions, window, skills, budget_str, mode, start, end):
    return single_exp(
        np.arange(start, end),
        [decisions],
        [window],
        [mode],
        skills,
        [budget_str],
    )


def str_to_float(s):
    if "/" not in s:
        return float(s)
    i, j = s.split("/")
    return float(i) / float(j)


if __name__ == "__main__":

    # decisions = int(sys.argv[1])
    # start, end = [int(i) for i in sys.argv[2].split("-")]
    # main_multi(decisions, start, end)

    if sys.argv[1] == "single":

        decisions = int(sys.argv[2])
        mode = sys.argv[3]
        budget_str = sys.argv[4]

        start, end = [int(i) for i in sys.argv[5].split("-")]
        window = 0

        skills = "ABCD"[:decisions]

        # skills_strs = [i for i in skills]  # Only 4.1, not 4.2

        baby, rev = [], []
        for i in range(1, decisions):
            baby.append(skills[:i])
        for i in range(decisions):
            baby.append(skills[i:])
            rev.append(skills[i:])
        onepass = "_".join(skills)
        baby = "_".join(baby)
        rev = "_".join(rev)
        skills_strs = [baby, rev, onepass] + [i for i in skills[1:]]

        print(skills_strs)

        main(decisions, window, skills_strs, budget_str, mode, start, end)

    if sys.argv[1] == "algo":

        decisions = int(sys.argv[2])
        mode = sys.argv[3]
        start_budget = int(sys.argv[4])
        end_budget = int(sys.argv[5])
        trial_inc = str_to_float(sys.argv[6])
        budget_inc = str_to_float(sys.argv[7])
        start, end = [int(i) for i in sys.argv[8].split("-")]
        window = 0

        skills = "ABCD"[:decisions]
        # skills = "AD"
        baby, rev = [], []
        for i in range(1, decisions):
            baby.append(skills[:i])
        for i in range(decisions):
            baby.append(skills[i:])
            rev.append(skills[i:])
        onepass = "_".join(skills)
        baby = "_".join(baby)
        rev = "_".join(rev)
        skills_strs = [baby]

        for seed in range(start, end):

            budget = np.ones(decisions) * start_budget / decisions
            budget = budget.astype(int)
            while sum(budget) < end_budget:
                cur = sum(budget)
                branch_add = np.ceil(cur * trial_inc).astype(int)
                branch_vals = {}
                print(f"{cur} - {budget}")
                print("{")
                for i in range(decisions):
                    branch_budget = budget.copy()
                    branch_budget[i] += branch_add
                    # valid_score, test_score = np.random.random(2).round(2)
                    loss, test_score = exp(
                        seed,
                        decisions,
                        window,
                        mode,
                        onepass,
                        branch_budget,
                        write=False,
                        verbose=False,
                    )
                    branch_vals[i] = (loss, test_score)
                    print(f"    {branch_budget}: {branch_vals[i]}")
                print("}")

                best_branch = max(branch_vals, key=lambda x: branch_vals[x][1])
                print(f"Best branch: {best_branch} - {branch_vals[best_branch]}")
                loss_algo, test_score_algo = branch_vals[best_branch]

                total_next_budget = (
                    np.ceil(cur * budget_inc / start_budget).astype(int) * start_budget
                )
                print(budget, total_next_budget)
                # budget += branch_add
                budget[best_branch] += total_next_budget - cur
                print(budget)

                _, test_score_chosen = exp(
                    seed,
                    decisions,
                    window,
                    mode,
                    onepass,
                    budget,
                    write=False,
                    verbose=False,
                )
                print(f"Chosen Split - {test_score_chosen:.2f}")

                _, test_score_equal = exp(
                    seed,
                    decisions,
                    window,
                    mode,
                    onepass,
                    np.ones(decisions) * sum(budget) // decisions,
                    write=False,
                    verbose=False,
                )
                print(f"Random Split - {test_score_equal:.2f}")

                _, test_score_high = exp(
                    seed,
                    decisions,
                    window,
                    mode,
                    onepass,
                    np.array([0 for _ in range(decisions - 1)] + [sum(budget)]),
                    write=False,
                    verbose=False,
                )
                print(f"High Split - {test_score_high:.2f}")

                with open("/storage1/fs1/chien-ju.ho/Active/tree/algo5.txt", "a") as f:
                    print(
                        f"{seed};{decisions};{mode};{start_budget};{end_budget};{trial_inc};{budget_inc};{best_branch};{branch_vals};{budget};{loss_algo};{test_score_algo};{test_score_chosen};{test_score_equal};{test_score_high}",
                        file=f,
                        flush=True,
                    )

                print("\n")
            print(f"{sum(budget)} - {budget}")
