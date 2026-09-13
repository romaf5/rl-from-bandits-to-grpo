"""Assemble the notebook from section modules. Usage: python assemble.py OUT.ipynb [sections...]"""
import sys, importlib, nbformat, copy, base64

ORDER = ["sec0_setup", "sec1_foundations", "sec2_bandits", "sec3_dp", "sec4_tabular",
         "sec5_dqn", "sec6_reinforce", "sec7_actor_critic", "sec8_trpo", "sec9_ppo",
         "sec10_llm", "sec11_reward_model", "sec12_ppo_llm", "sec13_dpo", "sec14_rloo",
         "sec15_grpo", "sec16_beyond_grpo", "sec17_agentic", "sec18_wrapup"]

def build(out, sections=None, solutions=False):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for name in (sections or ORDER):
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            print("skipping missing section", name); continue
        for c in mod.CELLS:
            c = copy.deepcopy(c)
            if solutions and "exercise_solution" in c.metadata:
                c.source = base64.b64decode(c.metadata["exercise_solution"]).decode()
            c.metadata.pop("exercise_solution", None)
            nb.cells.append(c)
    nbformat.write(nb, out)
    print(f"wrote {out} with {len(nb.cells)} cells")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--solutions"]
    build(args[0], args[1:] or None, solutions="--solutions" in sys.argv)
