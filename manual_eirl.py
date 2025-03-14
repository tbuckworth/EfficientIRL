import eirl
from helper_local import load_json


def main():
    cfg = load_json("hp_tune/bc_seals_hopper_best_hp_eval.json")

    expert_trainer = eirl.EIRL(**cfg["eirl"])



if __name__ == "__main__":
    main()
