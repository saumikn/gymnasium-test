import numpy as np
from tqdm import tqdm
import sys
from itertools import product
import gc
import time
from multiprocessing import Process
from pathlib import Path
from collections import defaultdict

# ruff: noqa: E402
from utils import import_tf

tf = import_tf()

tf.config.set_visible_devices([], "GPU")


def softmax(a, temp):
    a = a / temp
    a -= a.max()
    a = np.exp(a) / np.exp(a).sum()
    return a


def get_skill_key(skill: str, mode: str, depth: int):
    if skill.startswith("D-"):
        return (4, float(skill.split("-")[1]))
    elif skill.isupper():
        return ("ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(skill) + 1, 0)
    else:
        skill = skill.upper()
        # Read closest.txt to get the temperature for this mode/skill/depth combination
        with open("closest.txt", "r") as f:
            for line in f:
                if line.startswith(f"{depth},{mode},{skill},"):
                    temp = float(line.strip().split(",")[-1])
                    return (4, temp)
        raise ValueError(f"No temperature found for {skill} in {mode} at depth {depth}")


class Tree:
    def __init__(self, depth=10, mode="float", seed=0):
        self.depth = depth
        self.n = 2 ** (self.depth) - 1
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        self.get_all_path_cache = {}
        self.get_visible_arr_cache = {}
        if mode == "random":
            self.arr = self.rng.random(self.n)
        elif mode.startswith("crit-"):
            # a - cost to enter critical state
            # b - reward for correct choice in critical state
            # c - cost for incorrect choice in critical state
            # d - probability of critical state

            self.arr = self.rng.random(self.n)
            rewards = self.rng.random(self.n // 2)
            a, b, c, d = [float(i) for i in mode.split("-")[1:]]
            for i, reward in enumerate(rewards):
                _l = self._l(i)
                _r = self._r(i)
                if reward < d / 2:  # In reward state and reward is left node
                    self.arr[i] += -a
                    self.arr[_l] += b
                    self.arr[_r] += -c
                elif reward < d:  # In reward state and reward is right node
                    self.arr[i] += -a
                    self.arr[_l] += -c
                    self.arr[_r] += b
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
        elif mode == "binary":
            self.arr = self.rng.integers(
                0, 1, endpoint=True, size=self.n, dtype=np.int32
            )
        else:
            raise ValueError("Mode must be start with 'crit-")

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

        _l = _r = i
        for _ in range(vision):
            if Tree._l(_l) < self.n:
                _l = Tree._l(_l)
                _r = Tree._r(_r)

        paths = [self.get_upstream(node) for node in range(_l, _r + 1)]
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
        vision, temp = get_skill_key(skill, self.mode, self.depth)
        if Tree._l(i) >= self.n or vision < 0:
            return None
        lscore = max(self.get_scores(Tree._l(i), vision - 1))
        rscore = max(self.get_scores(Tree._r(i), vision - 1))

        scores = np.array([lscore, rscore])
        if temp > 0:
            scores = softmax(scores, temp)
            return self.rng.choice(2, p=scores)
        else:
            return int(np.argmax(scores))

    def get_path(self, i, skill, pertub=None):
        path = []
        row = 0
        while i < self.n:
            choice = self.get_choice(i, skill)
            if choice is not None and pertub is not None and pertub == row:
                choice = 1 - choice
            path += [(i, choice)]
            if choice is None:
                break
            i = Tree._l(i) + choice
            row += 1
        return path

    def get_total_score(self, skill, pertub=None):
        path = self.get_path(0, skill, pertub)
        return self.score_path([p[0] for p in path])

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

        # Extract policy actions
        Y_policy = np.array([choice for (_, choice) in path])

        # Calculate Y_value as array with values for each step in path
        if len(path) > 1:
            path_indices = [p for p, _ in path[1:]]
            path_values = np.array([self.arr[p] for p in path_indices])
            Y_value = np.sum(path_values)
        else:
            Y_value = 0

        return X, {"policy": Y_policy, "value": np.full((len(Y_policy), 1), Y_value)}

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
        X, Y_dict = [], {"policy": [], "value": []}
        for tree in self.trees:
            _x, _y_dict = tree.get_training_data(skill, window, leaf)
            X.append(_x)
            Y_dict["policy"].append(_y_dict["policy"])
            Y_dict["value"].append(_y_dict["value"])

        X_array = np.concatenate(X)
        Y_dict_array = {
            "policy": np.concatenate(Y_dict["policy"]),
            "value": np.concatenate(Y_dict["value"]),
        }
        del X
        del Y_dict
        return X_array, Y_dict_array

    def eval_model(self, model, window, temp=1):
        if window == 0:
            n = self.trees[0].n
            X = np.zeros((self.m, n * 2))
            for i, tree in enumerate(self.trees):
                X[i, :n] = tree.arr
            locs = np.zeros(self.m, dtype=np.int32)
            X[np.arange(self.m), locs + n] = 1

            paths = [locs]
            for i in range(self.depth - 1):
                Y = model(X)[0].numpy().argmax(axis=1)
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

                if temp == 0:
                    Y = model(X)[0].numpy().argmax(axis=1)
                    for i, choice in enumerate(Y):
                        paths[i].append(Tree._l(paths[i][-1]) + choice)
                else:
                    Y = model(X)[0].numpy()
                    for i, choice in enumerate(Y):
                        choice = softmax(choice, temp)
                        choice = np.random.default_rng().choice(2, p=choice)
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

        # Policy head (action probabilities)
        policy_logits = tf.keras.layers.Dense(2)(x)
        policy_output = tf.keras.layers.Softmax(name="policy")(policy_logits)

        # Value head (expected reward)
        value_output = tf.keras.layers.Dense(1, name="value")(x)

        model = tf.keras.models.Model(
            inputs=inputs, outputs=[policy_output, value_output]
        )

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

    # Define losses for both heads
    policy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    value_loss = tf.keras.losses.MeanSquaredError()

    # Compile with multiple losses
    model.compile(
        optimizer=opt,
        loss={"policy": policy_loss, "value": value_loss},
        loss_weights={"policy": 1.0, "value": 0},
        metrics={"policy": "accuracy"},
        run_eagerly=False,
    )
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
    opt="adam",
    lr=1,
    num_nodes=16,
    num_layers=1,
    verbose=True,
    perf_stopping=False,
    temp=1,
):
    depth = decisions + 1
    if verbose:
        print(f"\n\n\nStarting experiment: Seed-{seed} Depth-{depth} Skill-{skill_str}")

    st = time.perf_counter()
    model = make_model(depth, window, opt, lr, num_nodes, num_layers)
    best = model.get_weights()

    if decisions == 2 or False:
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

        elif budget.startswith("0_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0, 1])
        elif budget.startswith("25_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.25, 0.75])
        elif budget.startswith("50_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.5, 0.5])
        elif budget.startswith("75_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.75, 0.25])
        elif budget.startswith("100_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([1, 0])

        budget = budget.astype(int)
        budget = dict(zip("AD", budget))
        if verbose:
            print(budget)

    elif decisions == 4:
        if isinstance(budget, (list, np.ndarray)):
            budget = np.array(budget).astype(int)
            budget_str = "_".join(str(i) for i in budget)

        elif budget.startswith("right_"):
            budget_str = budget
            budget = int(budget_str.split("_")[1]) * np.array([0.125, 0.125, 0.25, 0.5])

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

    if verbose:
        print(
            f"make model: {seed} {depth} {skill_str} {budget_str}",
            time.perf_counter() - st,
        )

    skills = [[(s, budget[s.upper()]) for s in stage] for stage in skill_str.split("_")]
    if verbose:
        print(skills)

    print("iterating over stages:", skills)
    for stagei, stage in enumerate(skills):
        print("stage", stagei, stage)
        if verbose:
            print(stage)
        model = make_model(model=model)
        model.set_weights(best)

        X_trains, Y_trains, X_tests, Y_tests = [], [], [], []

        train_offset = int(1e8)
        test_offset = int(1e9)

        for si, (s, budget) in enumerate(stage):
            st = time.perf_counter()
            if not budget:
                continue

            # Calculate training and testing budgets for each skill
            if decisions == 4:
                train_budget = int(budget * 0.75)
                test_budget = int(budget * 0.25)
            else:
                train_budget = budget
                test_budget = 16384  # Default test budget for other cases

            train = Forest(
                depth,
                mode,
                train_budget * seed + si * train_offset,
                train_budget * (seed + 1) + si * train_offset,
            )
            X_train, Y_train = train.get_training_data(s, window=window)
            if verbose:
                print(
                    f"train data: {seed} {depth} {s} {train_budget}",
                    time.perf_counter() - st,
                )

            st = time.perf_counter()
            test = Forest(
                depth,
                mode,
                test_budget * seed + si * train_offset + test_offset,
                test_budget * (seed + 1) + si * train_offset + test_offset,
            )
            X_test, Y_test = test.get_training_data(s, window=window)
            if verbose:
                print(
                    f"test data: {seed} {depth} {s} {test_budget}",
                    time.perf_counter() - st,
                )
            X_trains.append(X_train)
            Y_trains.append(Y_train)
            X_tests.append(X_test)
            Y_tests.append(Y_test)

        if not X_trains:
            continue

        X_train = tf.convert_to_tensor(np.concatenate(X_trains))
        Y_train = {
            "policy": tf.convert_to_tensor(
                np.concatenate([y["policy"] for y in Y_trains])
            ),
            "value": tf.convert_to_tensor(
                np.concatenate([y["value"] for y in Y_trains])
            ),
        }
        X_test = tf.convert_to_tensor(np.concatenate(X_tests))
        Y_test = {
            "policy": tf.convert_to_tensor(
                np.concatenate([y["policy"] for y in Y_tests])
            ),
            "value": tf.convert_to_tensor(
                np.concatenate([y["value"] for y in Y_tests])
            ),
        }

        tqdm_kwargs = {
            "ncols": 80,
            "leave": True,
            "desc": f"Seed {seed} Depth {depth} Stage {'+'.join(s for s, _ in stage)}",
            # "position": 1,
            "disable": not verbose,
        }

        if perf_stopping:
            stopper = EarlyStopper(patience, minimize=False)
            best_test = -np.inf
        else:
            stopper = EarlyStopper(patience, minimize=True)
            best_test = np.inf

        for epoch in tqdm(range(10000), **tqdm_kwargs):
            # print()
            # st = time.perf_counter()
            if epoch != 0:
                model.fit(X_train, Y_train, batch_size=batch_size, verbose=0)
            # print(f"model fit: {seed} {epoch}", time.perf_counter() - st)

            # st = time.perf_counter()
            eval_result = model.evaluate(X_test, Y_test, batch_size=16384, verbose=0)
            total_loss, _, _, policy_acc = eval_result

            if perf_stopping:
                st = time.perf_counter()
                _, scores, _ = test.eval_model(model, window=window)
                valid_score = np.mean(scores[: test_budget // 2])
                test_score = np.mean(scores[test_budget // 2 :])
            else:
                valid_score, test_score = 0, 0
            # print(f"model eval_model: {seed} {epoch}", time.perf_counter() - st)
            at_best, at_patience = stopper.should_stop(total_loss)
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
                    batch_size,
                    opt,
                    lr,
                    num_nodes,
                    num_layers,
                    epoch,
                    total_loss,
                    policy_acc,
                    temp,
                    valid_score,
                    stopper.wait,
                    stopper.patience,
                    test_score,
                ]
                # print("at best:", get_output(best_output))

            if at_patience:
                print("at patience")
                if not perf_stopping:
                    model.set_weights(best)
                    _, scores, _ = test.eval_model(model, window=window, temp=temp)
                    best_test = np.mean(scores)
                    best_output[-1] = best_test

                if write:
                    print("writing: ", get_output(best_output))
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
    # filepath = "/storage1/fs1/chien-ju.ho/Active/tree/output30.txt"
    filepath = "/home/n.saumik/gymnasium-test/output/output38.txt"
    if not Path(filepath).exists():
        with open(filepath, "w") as f:
            print(
                "seed,depth,window,mode,stagei,skill_str,budget_str,batch_size,opt,lr,num_nodes,num_layers,epoch,loss,acc,temp,valid_score,wait,patience,test_score",
                file=f,
                flush=True,
            )
    with open(filepath, "a") as f:
        # with open("/home/n.saumik/gymnasium-test/output/output29.txt", "a") as f:
        print(output, file=f, flush=True)
        print(f"finished writing output ~~~{output}~~~to file={filepath}")


# @profile
def single_exp(seeds, decisions, windows, modes, skills, budgets, batch_sizes):
    tqdm_kwargs = {"ncols": 80, "leave": True, "disable": True}

    combos = list(
        product(seeds, decisions, windows, modes, skills, budgets, batch_sizes)
    )
    print(len(combos))
    for combo in tqdm(combos, **tqdm_kwargs):
        p = Process(target=exp, args=combo)
        p.start()
        p.join()
        # exp(*combo)

    # print(combos)

    # tqdm_kwargs = {"ncols": 80, "leave": True, "disable": True, "max_workers": 1}
    # process_map(exp, combos, **tqdm_kwargs)


def main(decisions, window, skills, budget_str, batch_size, mode, start, end):
    return single_exp(
        np.arange(start, end),
        [decisions],
        [window],
        [mode],
        skills,
        [budget_str],
        [batch_size],
    )


def str_to_float(s):
    if "/" not in s:
        return float(s)
    i, j = s.split("/")
    return float(i) / float(j)


if __name__ == "__main__":
    if sys.argv[1] == "single":
        decisions = int(sys.argv[2])  # depth of the tree minus 1
        mode = sys.argv[3]  # define the critical state generation type
        budget_str = sys.argv[4]  # how much training budget to use
        batch_size = int(sys.argv[5])

        start, end = [int(i) for i in sys.argv[6].split("-")]
        window = 0

        # skills_strs = ["A_AD", "AD_D", "D"]
        if decisions == 2:
            skills_strs = ["A", "D", "a"]
        elif decisions == 4:
            skills_strs = ["A", "B", "C", "D", "a", "b", "c"] + [
                "A_AB_ABC_ABCD",
                "ABCD_BCD_CD_D",
            ]
        else:
            raise ValueError(f"Decisions {decisions} not implemented")

        # skills_strs = [baby]

        print(f"{skills_strs=}, {budget_str=}, {batch_size=}")

        main(decisions, window, skills_strs, budget_str, batch_size, mode, start, end)
