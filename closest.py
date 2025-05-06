import numpy as np
from tqdm import tqdm
import tree
import multiprocessing as mp
from pathlib import Path
from itertools import product


def get_closest(
    target, depth, mode, skill, forest_size, lower_bound, upper_bound, step, write=False
):
    f = tree.Forest(depth, mode, 0, forest_size)

    closest, diff, res = 0, float("inf"), 0
    for temp in tqdm(
        np.arange(lower_bound, upper_bound, step),
        desc=f"{mode} {skill} (size={forest_size})",
        disable=True,
    ):
        temp = round(temp, 2)  # Round to 2 decimal places
        scores = [t.get_total_score(f"D-{temp}") for t in f.trees]
        _score = np.mean(scores)
        _diff = abs(target - _score)
        if _diff < diff:
            diff = _diff
            closest = temp

    result = (depth, mode, skill, closest)
    if write:
        write_result(result)
    return result


def get_reference_score(depth, mode, skill):
    """Get the reference score for a skill with a large forest."""
    print(f"Getting reference score for {mode} {skill}")
    f = tree.Forest(depth, mode, 0, 100000)
    scores = [t.get_total_score(skill) for t in f.trees]
    truth = np.mean(scores)
    # result = (depth, mode, skill, truth)
    # write_result(result)
    return truth


def write_result(result, filename="/home/n.saumik/gymnasium-test/closest.txt"):
    """Write a result to the output file."""
    depth, mode, skill, closest = result

    filepath = Path(filename)
    # Create the file with header if it doesn't exist
    if not filepath.exists():
        with open(filepath, "w") as f:
            print(
                "depth,mode,skill,closest_temp",
                file=f,
                flush=True,
            )

    # Append the result
    with open(filepath, "a") as f:
        line = f"{depth},{mode},{skill},{closest:.2f}"
        print(line, file=f, flush=True)

    # Also print to console
    print(f"[{mode}] {skill} (depth={depth}): D-{closest:.2f}")


def process_mode_skill(args):
    """Process a single mode/skill combination through all forest sizes."""
    depth, mode, skill = args
    try:
        # First get the reference score with a large forest
        truth = get_reference_score(depth, mode, skill)

        # Phase 1: Broad search with small forest
        _, _, _, c1 = get_closest(truth, depth, mode, skill, 100, 0, 10, 0.1)

        # Phase 2: Narrower search with medium forest
        lower = max(c1 - 1, 0)
        upper = c1 + 1
        _, _, _, c2 = get_closest(truth, depth, mode, skill, 1000, lower, upper, 0.01)

        # Phase 3: Fine search with large forest
        lower = max(c2 - 0.5, 0)
        upper = c2 + 0.5
        _, _, _, c3 = get_closest(truth, depth, mode, skill, 10000, lower, upper, 0.01)

        # Phase 4: Final refinement with largest forest
        lower = max(c3 - 0.1, 0)
        upper = c3 + 0.1
        get_closest(truth, depth, mode, skill, 100000, lower, upper, 0.01, write=True)

        # write_result(final_result)

        # print(f"Completed {mode} {skill} with final temp {final_result[3]:.2f}")
        # return final_result
    except Exception as e:
        print(f"Error processing {mode} {skill}: {str(e)}")
        return None


if __name__ == "__main__":
    depths = [3, 5, 9]

    modes = [
        "binary",
        "random",
        "linear-2",
        "linear-3",
        "linear-4",
        "exponential-2",
        "exponential-3",
        "exponential-4",
        "crit-0-2-8-0.2",
        "crit-0-2-8-0.4",
        "crit-0-2-8-0.6",
        "crit-0-2-8-0.8",
        "crit-0-2-8-1.0",
        "crit-8-10-0-0.2",
        "crit-8-10-0-0.4",
        "crit-8-10-0-0.6",
        "crit-8-10-0-0.8",
        "crit-8-10-0-1.0",
    ]

    skills = ["A", "B", "C"]

    # Create all combinations of modes and skills
    combinations = list(product(depths, modes, skills))

    # Progress tracking
    print(f"Processing {len(combinations)} mode-skill combinations...")

    # Start a pool of workers
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} processes")

    # Process all combinations in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_mode_skill, combinations)

    # Summarize results
    print("\nFinal Results Summary:")
    for result in results:
        if result:
            mode, skill, _, target, temp, score, diff = result
            print(f"{mode} {skill}: D-{temp:.2f} (diff={diff:.6f})")

    print("\nAll searches completed. Results saved to output/closest.txt")
