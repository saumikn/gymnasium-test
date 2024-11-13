print("importing numpy")

import numpy as np
from tqdm import tqdm
import sys
from itertools import product
import os
import gc
import time
import pathlib

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
    "E": (5, 0),
    "F": (6, 0),
}


class Tree:
    def __init__(self, depth=10, mode="float", seed=0):
        self.depth = depth
        self.n = 2 ** (self.depth) - 1
        self.mode = mode
        self.get_all_path_cache = {}
        self.get_visible_arr_cache = {}
        self.rng = np.random.default_rng(seed)
        if mode.startswith("crit-"):
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
            raise ValueError("Mode must start with 'crit-")

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
    depth=None, window=0, opt="adam", lr=1, num_nodes=16, num_layers=1, model=None
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


def get_budget_str(decisions, budget):
    if decisions in [3, 5]:
        if isinstance(budget, (list, np.ndarray)):
            budget = np.array(budget).astype(int)
            budget_str = "_".join(str(i) for i in budget)

        elif budget.startswith("right_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([1/6,1/3,1/2])  # fmt: skip

        elif budget.startswith("equal_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([1/3,1/3,1/3])  # fmt: skip

        elif budget.startswith("left_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([1/2,1/3,1/6])  # fmt: skip

        budget = budget.astype(int)
        if decisions == 5:
            budget = dict(zip("ACE", budget))
        elif decisions == 3:
            budget = dict(zip("ABC", budget))

    if decisions == 4:
        if isinstance(budget, (list, np.ndarray)):
            budget = np.array(budget).astype(int)
            budget_str = "_".join(str(i) for i in budget)

        elif budget.startswith("right_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array(
                [1 / 16, 3 / 16, 5 / 16, 7 / 16]
            )

        elif budget.startswith("equal_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.25, 0.25, 0.25, 0.25])

        elif budget.startswith("left_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array(
                [7 / 16, 5 / 16, 3 / 16, 1 / 16]
            )

        budget = budget.astype(int)
        budget = dict(zip("ABCD", budget))

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
    return budget, budget_str


def train_model(
    perf_stopping,
    patience,
    model,
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    test_forest,
    window,
):
    tqdm_kwargs = {
        "ncols": 80,
        "leave": True,
        # "desc": f"Seed {seed} Depth {depth} Stage {'+'.join(s for s, _ in stage)}",
        # "position": 1,
        "disable": False,
    }
    if perf_stopping:
        stopper = EarlyStopper(patience, minimize=False)
    else:
        stopper = EarlyStopper(patience, minimize=True)

    stopper.wait = 0
    for epoch in tqdm(range(10000), **tqdm_kwargs):
        if epoch != 0:
            model.fit(X_train, Y_train, batch_size=32, verbose=0)
        loss, acc = model.evaluate(
            X_valid, Y_valid, batch_size=min(16384, len(X_valid)), verbose=0
        )
        # print(f"{epoch}: {loss:.3f} {stopper.wait}")

        at_best, at_patience = stopper.should_stop(loss)
        if at_best or acc == 1:
            best = [model.get_weights(), loss, acc, 0]

        if at_patience or acc == 1:
            model.set_weights(best[0])
            _, scores, _ = test_forest.eval_model(model, window=window)
            best[-1] = np.mean(scores)
            print(f"Best - {best[-1]:.3f}")
            return best


def get_2level_XY(seed, skills, depth, window):
    train_offset, test_offset, test_budget = int(1e8), int(1e9), 16384
    test_forest = Forest(
        depth,
        mode,
        test_budget * seed + test_offset,
        test_budget * (seed + 1) + test_offset,
    )
    s0, b0 = skills[0][0]
    train_forest0 = Forest(
        depth,
        mode,
        b0 * seed + 0 * train_offset,
        b0 * (seed + 1) + 0 * train_offset,
    )
    X0, Y0 = train_forest0.get_training_data(s0, window=window)

    if len(skills[0]) == 2:
        s1, b1 = skills[0][1]
        train_forest1 = Forest(
            depth,
            mode,
            b1 * seed + 1 * train_offset,
            b1 * (seed + 1) + 1 * train_offset,
        )
        X1, Y1 = train_forest1.get_training_data(s1, window=window)
    else:
        X1, Y1 = None, None

    return X0, Y0, X1, Y1, test_forest


def get_XY(seed, stage, depth, mode, window, offset=int(1e8), test_budget=16384):
    o0, o1, o2 = offset, offset * 2, offset * 3
    s, b = stage
    b0, b1 = int(b * 0.75), int(b * 0.25)
    seed1 = seed + 1
    train_forest = Forest(
        depth,
        mode,
        b0 * seed + o0,
        b0 * seed1 + o0,
    )
    valid_forest = Forest(
        depth,
        mode,
        b1 * seed + o1,
        b1 * seed1 + o1,
    )
    test_forest = Forest(
        depth,
        mode,
        test_budget * seed + o2,
        test_budget * seed1 + o2,
    )
    X0, Y0 = train_forest.get_training_data(s, window=window)
    X1, Y1 = valid_forest.get_training_data(s, window=window)
    return (X0, Y0), (X1, Y1), test_forest


def exp(
    seed,
    decisions,
    mode,
    skill_str,
    budget,
    discrim_thresholds,
    discrim_steps,
    window=0,
    patience=20,
    opt="adam",
    lr=1,
    num_nodes=64,
    num_layers=3,
    verbose=True,
    perf_stopping=False,
):
    depth = decisions + 1
    if verbose:
        print(f"\n\n\nStarting experiment: Seed-{seed} Depth-{depth} Skill-{skill_str}")

    budget, budget_str = get_budget_str(decisions, budget)
    skills = [[(s, budget[s]) for s in stage] for stage in skill_str.split("_")]

    data = {}
    for stage in skills:
        for s, b in stage:
            if s in data:
                continue
            data[s] = get_XY(seed, (s, b), depth, mode, window)

    for t in discrim_thresholds:
        model = make_model(depth, window, opt, lr, num_nodes, num_layers)
        best = model.get_weights()
        for stagei, stage in enumerate(skills):
            X_trains, Y_trains, X_valids, Y_valids = [], [], [], []
            for s, b in stage:
                (Xt, Yt), (Xv, Yv), test_forest = data[s]
                X_trains.append(Xt)
                Y_trains.append(Yt)
                X_valids.append(Xv)
                Y_valids.append(Yv)
            if not X_trains:
                continue
            X_trains = tf.convert_to_tensor(np.concatenate(X_trains))
            Y_trains = tf.convert_to_tensor(np.concatenate(Y_trains))
            X_valids = tf.convert_to_tensor(np.concatenate(X_valids))
            Y_valids = tf.convert_to_tensor(np.concatenate(Y_valids))

            stopper = EarlyStopper(patience, minimize=True)
            model = make_model(model=model)
            model.set_weights(best)
            tqdm_kwargs = {
                "ncols": 80,
                "leave": True,
                "desc": f"Seed {seed} Depth {depth} Stage {'+'.join(s for s, _ in stage)}",
                # "position": 1,
                "disable": not verbose,
            }
            for epoch in tqdm(range(10000), **tqdm_kwargs):
                if epoch != 0:
                    model.fit(X_trains, Y_trains, batch_size=64, verbose=0)
                loss, acc = model.evaluate(
                    X_valids, Y_valids, batch_size=16384, verbose=0
                )
                at_best, at_patience = stopper.should_stop(loss)
                if at_best:
                    best = model.get_weights()
                    best_output = [seed, depth - 1, window, mode, stagei, skill_str, budget_str, t, discrim_steps, opt, lr, num_nodes, num_layers, loss, acc, 0]  # fmt: skip
                if at_patience:
                    if perf_stopping == False:
                        model.set_weights(best)
                        _, scores, _ = test_forest.eval_model(model, window=window)
                        best_test = np.mean(scores)
                        best_output[-1] = best_test
                    write_output(get_output(best_output))
                    break

    return

    assert len(skills) == 1

    if len(skills[0]) == 1:
        model = make_model(depth, window, opt, lr, num_nodes, num_layers)
        X0, Y0, X1, Y1, test_forest = get_2level_XY(seed, skills, depth, window)
        train_size = int(len(X0) * 0.75)
        X_train = tf.convert_to_tensor(X0[:train_size])
        Y_train = tf.convert_to_tensor(Y0[:train_size])
        X_valid = tf.convert_to_tensor(X0[train_size:])
        Y_valid = tf.convert_to_tensor(Y0[train_size:])

        _, loss, acc, test = train_model(
            perf_stopping,
            patience,
            model,
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            test_forest,
            window,
        )
        best_output = [
            seed,
            depth - 1,
            window,
            mode,
            skill_str,
            budget_str,
            0,
            0,
            opt,
            lr,
            num_nodes,
            num_layers,
            loss,
            acc,
            test,
        ]
        write_output(get_output(best_output))

    elif len(skills[0]) == 2:
        X0, Y0, X1, Y1, test_forest = get_2level_XY(seed, skills, depth, window)

        print(X0.shape, Y0.shape, X1.shape, Y1.shape)

        discriminator = make_model(depth, window, opt, lr, num_nodes, num_layers)
        discX = np.concatenate([X0, X1])
        discY = np.concatenate([np.ones_like(Y0), np.zeros_like(Y1)])
        print(discX.shape, discY.shape)
        ds = tf.data.Dataset.from_tensor_slices((discX, discY))
        ds = ds.repeat().shuffle(100000).batch(32).take(discrim_steps)
        acc = discriminator.fit(ds, verbose=1).history["accuracy"][0]
        low_pred = discriminator(X1).numpy()

        print(low_pred.shape)
        print(":" * 100)

        last_mask = None
        last_res = None

        for t in discrim_thresholds:
            print(f"Discrim Threshold - {t}")
            model = make_model(depth, window, opt, lr, num_nodes, num_layers)

            if t.startswith("p-"):
                t2 = float(t.split("-")[1])
                mask = low_pred[:, 1] <= np.quantile(low_pred[:, 1], t2)
            elif t.startswith("p+"):
                t2 = float(t.split("+")[1])
                mask = low_pred[:, 1] >= np.quantile(low_pred[:, 1], t2)
            elif t.startswith("r-"):
                t2 = float(t.split("-")[1])
                mask = np.zeros_like(low_pred[:, 1])
                mask[: int(t2 * len(mask))] = 1
                mask = mask.astype(bool)
            else:
                mask = (
                    low_pred[:, 1] <= t
                )  # Probability of being high skill is less than t
            print(f"Num of matches: {mask.sum()}")

            if sum(mask) == last_mask:
                print("No change from last threshold, using last output")
                loss, acc, test = last_res

            else:
                X_all = np.concatenate([X0, X1[mask]])
                Y_all = np.concatenate([Y0, Y1[mask]])
                print(X_all.shape, Y_all.shape)

                train_size = int(len(X_all) * 0.75)
                print(train_size)
                X_train = tf.convert_to_tensor(X_all[:train_size])
                Y_train = tf.convert_to_tensor(Y_all[:train_size])
                X_valid = tf.convert_to_tensor(X_all[train_size:])
                Y_valid = tf.convert_to_tensor(Y_all[train_size:])

                print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

                _, loss, acc, test = train_model(
                    perf_stopping,
                    patience,
                    model,
                    X_train,
                    Y_train,
                    X_valid,
                    Y_valid,
                    test_forest,
                    window,
                )
                last_mask = sum(mask)
                last_res = loss, acc, test

            best_output = [
                seed,
                depth - 1,
                window,
                mode,
                skill_str,
                budget_str,
                t,
                discrim_steps,
                opt,
                lr,
                num_nodes,
                num_layers,
                loss,
                acc,
                test,
            ]
            write_output(get_output(best_output))
            # print(get_output(best_output))
            print("-" * 50)


def get_output(args):
    return ",".join([str(a) for a in args])


def write_output(output):
    OUTPUT_PATH = "/scratch1/fs1/chien-ju.ho/Active/tree/output38.txt"
    if not pathlib.Path(OUTPUT_PATH).is_file():
        with open(OUTPUT_PATH, "a") as f:
            print(
                "seed,decisions,window,mode,stagei,skills,budget,disc_thresh,discrim_steps,opt,lr,nodes,layers,loss,acc,test",
                file=f,
                flush=True,
            )

    with open(OUTPUT_PATH, "a") as f:
        print(output, file=f, flush=True)


# @profile
def single_exp(*args):

    tqdm_kwargs = {"ncols": 80, "leave": True, "disable": True}

    combos = list(product(*args))
    print(len(combos))
    for combo in tqdm(combos, **tqdm_kwargs):
        exp(*combo)


def main(
    start, end, decisions, mode, skills, budget_str, discrim_thresholds, discrim_steps
):
    return single_exp(
        np.arange(start, end),
        [decisions],
        [mode],
        skills,
        [budget_str],
        [discrim_thresholds],
        [discrim_steps],
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

        start, end = [int(i) for i in sys.argv[2].split("-")]
        decisions = int(sys.argv[3])
        mode = sys.argv[4]
        budget_str = sys.argv[5]
        discrim_steps = int(sys.argv[6])
        # discrim_thresholds = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1]
        # discrim_thresholds = ['p-0', 'p-0.2', 'p-0.4', 'p-0.6', 'p-0.8', 'p-1', 'p+0', 'p+0.2', 'p+0.4', 'p+0.6', 'p+0.8', 'p+1']
        # discrim_thresholds = [
        #     "r-0",
        #     "r-0.01",
        #     "r-0.05",
        #     "r-0.1",
        #     "r-0.2",
        #     "r-0.4",
        #     "r-0.6",
        #     "r-0.8",
        #     "r-0.9",
        #     "r-0.95",
        #     "r-0.99",
        #     "r-1",
        # ]
        discrim_thresholds = [1]

        # skills = "ABCD"[:decisions]
        # skills_strs = [i for i in skills]  # Only 4.1, not 4.2
        # baby, rev = [], []
        # for i in range(1, decisions):
        #     baby.append(skills[:i])
        # for i in range(decisions):
        #     baby.append(skills[i:])
        #     rev.append(skills[i:])
        # onepass = "_".join(skills)
        # baby = "_".join(baby)
        # rev = "_".join(rev)
        # skills_strs = [baby, rev, onepass] + [i for i in skills[1:]]

        if decisions == 3:
            skills_strs = [
                "C",
                "B",
                "A_C",
                "CA_C",
                "A_CA_C",
                "CBA_CB_C",
                "A_B_C",
                "A_BA_CBA_CB_C",
            ]
            print(skills_strs)
        if decisions == 4:
            skills_strs = [
                "D",
                "C",
                "B",
                "A_D",
                "DA_D",
                "A_DA_D",
                "DCBA_DCB_DC_D",
                "A_B_C_D",
                "A_BA_CBA_DCBA_DCB_DC_D",
            ]
            print(skills_strs)
        elif decisions == 5:
            skills_strs = [
                "E",
                "C",
                "A_E",
                "EA_E",
                "A_EA_E",
                "ECA_EC_E",
                "A_C_E",
                "A_CA_ECA_EC_E",
            ]
            # skills_strs = ["EA"]
            print(skills_strs)
        main(
            start,
            end,
            decisions,
            mode,
            skills_strs,
            budget_str,
            discrim_thresholds,
            discrim_steps,
        )

    if sys.argv[1] == "algo":

        decisions = int(sys.argv[2])
        mode = sys.argv[3]
        start_budget = int(sys.argv[4])
        end_budget = int(sys.argv[5])
        trial_inc = str_to_float(sys.argv[6])
        budget_inc = str_to_float(sys.argv[7])
        start, end = [int(i) for i in sys.argv[8].split("-")]
        window = 0

        # skills = "ABCD"[:decisions]
        skills = "AD"
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

                with open("algo5.txt", "a") as f:
                    print(
                        f"{seed};{decisions};{mode};{start_budget};{end_budget};{trial_inc};{budget_inc};{best_branch};{branch_vals};{budget};{loss_algo};{test_score_algo};{test_score_chosen};{test_score_equal};{test_score_high}",
                        file=f,
                        flush=True,
                    )

                print("\n")
            print(f"{sum(budget)} - {budget}")
