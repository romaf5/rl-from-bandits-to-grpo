# Notebook builder

The notebook is generated from these section files (one `secNN_*.py` per Part) so it can be edited and
regenerated cleanly.

**Spoiler warning:** the section files contain the quiz answer keys and exercise solutions in plain text.
Don't read them if you want to take the quizzes honestly.

```bash
pip install nbformat nbclient ipykernel     # plus numpy, matplotlib, torch
python assemble.py ../../rl_from_bandits_to_grpo.ipynb            # build (no outputs)
python run_nb.py ../../rl_from_bandits_to_grpo.ipynb ../../rl_from_bandits_to_grpo.ipynb   # execute in place
python assemble.py solutions.ipynb --solutions                    # variant with reference solutions filled in
```
