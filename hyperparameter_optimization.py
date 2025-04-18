import re
import time
import traceback
from math import floor, log10

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from gp import bayesian_optimisation
from helper_local import filter_params

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    wandb = wandb_login = None
    pass


def get_wandb_performance(hparams, project="EfficientIRL", id_tag="sa_rew",
                          opt_metric="summary.original_ep_return_mean",
                          entity="ic-ai-safety"):
    if not isinstance(id_tag, list):
        id_tag = [id_tag]
    wandb_login()
    api = wandb.Api()
    entity, project = entity, project
    runs = api.runs(entity + "/" + project,
                    # filters={"$and": [{"tags": id_tag, "state": "finished"}]}
                    filters={"$and": [{"tags": {"$in": id_tag}}, {"state": "finished"}]}
                    )

    summary_list, config_list, name_list, state_list = [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        all_dicts.append(s_dict)

    # maybe this should just be DataFrame(all_dicts) - test
    df = pd.DataFrame.from_dict(all_dicts)
    try:
        y = df[opt_metric]
    except KeyError:
        return None, None

    if y.ndim==2:
        flt = np.bitwise_and(pd.notna(y).all(axis=1), (y != 'NaN').all(axis=1))
    elif y.ndim==1:
        flt = np.bitwise_and(pd.notna(y), (y != 'NaN'))
    else:
        raise NotImplementedError("ndim should be max 2")

    df = df[flt]
    y = y[flt]
    if len(df) == 0:
        return None, None
    # hp = [x for x in df.columns if re.search("config", x)]
    # hp = [h for h in hp if h not in ["config.extra_tags"]]
    # hp = [h for h in hp if len(df[h].unique()) > 1]

    if hparams is None:
        hp = [column for column in df.columns if re.search(fr"config\.",column)]
        return df[hp], y
    hp = [f"config.{h}" for h in hparams if f"config.{h}" in df.columns]
    dfn = df[hp].select_dtypes(include='number')
    return dfn, y


def n_sig_fig(x, n):
    if x == 0 or isinstance(x, bool):
        return x
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def select_next_hyperparameters(X, y, bounds, greater_is_better=True):
    if bounds == {}:
        return {}
    [b.sort() for b in bounds.values()]

    if X is None:
        bound_array = np.array([[x[0], x[-1]] for x in bounds.values()])
        next_params = np.random.uniform(bound_array[:, 0], bound_array[:, 1], (bound_array.shape[0]))
        col_order = bounds.keys()
    else:
        col_order = [re.sub(r"config\.", "", k) for k in X.columns]
        bo = [bounds[k] for k in col_order]

        bound_array = np.array([[x[0], x[-1]] for x in bo])

        xp = X.to_numpy()
        yp = y.to_numpy()

        params = []
        idx = np.random.permutation(len(X.columns))

        n_splits = np.ceil(len(idx) / 2)
        xs = np.array_split(xp[:, idx], n_splits, axis=1)
        bs = np.array_split(bound_array[idx], n_splits, axis=0)

        for x, b in zip(xs, bs):
            # Change to random search if problem
            param = bayesian_optimisation(x, yp, b, random_search=False, greater_is_better=greater_is_better)
            params += list(param)

        next_params = np.array(params)[np.argsort(idx)]

    int_params = [np.all([isinstance(x, int) for x in bounds[k]]) for k in col_order]
    bool_params = [np.all([isinstance(x, bool) for x in bounds[k]]) for k in col_order]
    next_params = [int(round(v, 0)) if i else v for i, v in zip(int_params, next_params)]
    next_params = [bool(v) if b else v for b, v in zip(bool_params, next_params)]

    hparams = {k: n_sig_fig(next_params[i], 3) for i, k in enumerate(col_order)}

    return hparams


def run_next_hyperparameters(hparams):
    from train_EIRL import trainEIRL
    filtered_params = filter_params(hparams, trainEIRL)
    trainEIRL(**filtered_params)


def run_next_hyperparameters_imeow(hparams):
    from train_IMEow import trainIMEow
    filtered_params = filter_params(hparams, trainIMEow)
    trainIMEow(**filtered_params)


def get_project(env_name, exp_name):
    # TODO: make this real
    return "EfficientIRL"


from sklearn.tree import _tree


def extract_rules(X, flt, tree, feature_names):
    tree_ = tree.tree_

    def recurse(X, flt, node, current_rule):
        # Not a leaf node.
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            f = X[name] <= threshold
            if flt[f].mean() > 0.5:
                current_rule += [(name, "<=", threshold)]
                recurse(X, flt, tree_.children_left[node], current_rule)
            else:
                current_rule += [(name, ">", threshold)]
                recurse(X, flt, tree_.children_right[node], current_rule)
        return current_rule
        #
        # else:
        #     # For a leaf, you might check if the prediction is "good"
        #     # and then add the rule if it is.
        #     # (Assume a binary classifier with positive class at index 1)
        #     if tree.value[node][0, 1] > tree.value[node][0, 0]:
        #         rules.append(current_rule)

    rules = recurse(X, flt,0, [])
    return rules

def tree_analyze_hparams(id_tag,
                         project="EfficientIRL",
                         ):
    from sklearn.tree import export_text

    opt_metric = ["summary.reward/original_ep_return_mean","summary.eirl/reward_correl"]

    X, y = get_wandb_performance(None, project, id_tag, opt_metric)


    X_clean = format_df_to_numbers(X, remove=["config.seed"])
    flt = np.bitwise_and(
        y["summary.reward/original_ep_return_mean"] > 350,
        y["summary.eirl/reward_correl"] > 0.5
    )
    up = (~flt).sum()/flt.sum()
    sample_weight = np.ones_like(flt)
    sample_weight[flt] = up

    # X_train, X_test, y_train, y_test = train_test_split(X, flt, test_size=0.2)
    tree = DecisionTreeClassifier(random_state=0)
    # tree = DecisionTreeClassifier(
    #     random_state=0,
    #     class_weight='balanced',
    #     criterion='entropy',
    #     min_impurity_decrease=0.0001  # Lower threshold for splitting
    # )
    tree.fit(X_clean, flt, sample_weight=sample_weight)
    print(export_text(tree, feature_names=list(X_clean.columns)))
    rules = extract_rules(X, flt, tree, list(X_clean.columns))

    f_all = np.ones(len(X_clean))
    for key, direction, value in rules:
        if direction == "<=":
            f = X_clean[key]<=value
        else:
            f = X_clean[key]>value
        f_all = np.bitwise_and(f_all, f)
        m = y[f_all]
        print("\nMean:")
        print(m.mean())
        # print("Std:")
        # print(m.std())

    y[f_all]

    print("done")

def format_df_to_numbers(X, remove=None):
    if remove is None:
        remove = []
    df = X.copy()

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Apply one-hot encoding to object (and string) columns
    object_cols = df.select_dtypes(include=['object']).columns
    non_list_cols = [col for col in object_cols if not df[col].apply(lambda x: isinstance(x, list) or isinstance(x, dict)).any()]
    final_cols = [col for col in non_list_cols if X[col].nunique() < 10]
    # One-hot encode only non-list object columns
    one_hots = pd.get_dummies(df[final_cols], columns=final_cols, drop_first=False).astype(int)

    numerics = X.select_dtypes(include=['float','int','bool'])

    df = pd.concat((numerics, one_hots),axis=1)
    columns = [col for col in df.columns if col not in remove]
    return df[columns]

def optimize_hyperparams(bounds,
                         fixed,
                         project="EfficientIRL",
                         id_tag="sa_rew",
                         run_next=run_next_hyperparameters,
                         opt_metric="summary.original_ep_return_mean",
                         greater_is_better=True,
                         abs=False,
                         debug=False,
                         ):
    strings = {k: v for k, v in fixed.items() if isinstance(v, list) and k != "extra_tags"}
    string_select = {k: v[np.random.choice(len(v))] for k, v in strings.items()}
    if "env_name" in string_select.keys():
        project = "EfficientIRL"
    try:
        X, y = get_wandb_performance(bounds.keys(), project, id_tag, opt_metric)
        if abs:
            y = y.abs()
    except ValueError as e:
        print(f"Error from wandb:\n{e}\nPicking hparams randomly.")
        X, y = None, None

    if X is not None and np.prod(X.shape) == 0:
        X, y = None, None

    hparams = select_next_hyperparameters(X, y, bounds, greater_is_better)

    fh = fixed.copy()
    hparams.update(fh)
    hparams.update(string_select)

    hparams = {k: int(v) if isinstance(v, np.int64) else v for k, v in hparams.items()}
    if debug:
        run_next(hparams)
    else:
        try:
            run_next(hparams)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            wandb.finish(exit_code=-1)


def init_wandb(cfg, prefix="symbolic_graph"):
    name = np.random.randint(int(1e5))
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    project = "EfficientIRL"
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
               tags=cfg["extra_tags"], resume=wb_resume, name=f"{prefix}-{name}")


def run_forever(bounds, fixed, run_func, opt_metric, abs=False, debug=False):
    project = "EfficientIRL"
    id_tag = fixed["extra_tags"][0]
    fixed["original_start"] = time.asctime()
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_func, opt_metric, greater_is_better=True, abs=abs,
                             debug=debug)


def search_eirl():
    fixed = dict(
        algo="eirl",
        seed=[0, 42, 100, 532, 3432],
        hard=False,
        reward_type=["next state only", "state"],
        log_prob_adj_reward=False,
        neg_reward=False,
        maximize_reward=False,
        rew_const=False,
        training_increments=5,
        extra_tags=["hopper_hp0","eirl"],
        early_learning=False,
        env_name="seals:seals/Hopper-v1",
        enforce_rew_val_consistency=False,
        gamma=0.98,
        batch_size=256,
        net_arch=[[256, 256, 256, 256]],
        consistency_coef=30.,
        n_envs=16,
        n_expert_demos=10,
        # lr=0.0005,
        l2_weight=0.001,
        flip_cartpole_actions=True,
    )
    bounds = dict(
        # consistency_coef=[2, 100.],
        n_epochs=[30, 150],
        # n_envs = [8, 32],
        # # log_prob_adj_reward=False,
        # # neg_reward=False,
        # # maximize_reward=False,
        # n_expert_demos=[1, 10],
        learner_timesteps=[250_000, 1000_000],
        # # gamma=[0.98],#[0.8, 0.999],
        # # training_increments=[5, 10],
        lr=[0.00025, 0.001],
        # l2_weight=[0, 0.002],
        # batch_size=[48, 96],
        # n_envs=[16, 48],
        # enforce_rew_val_consistency=False,
    )
    # bounds.update(fixed)
    # tree_analyze_hparams(id_tag=["disentanglement test", "disentanglement test2"],
    #                      project="EfficientIRL",
    #                      )
    run_forever(bounds, fixed, run_next_hyperparameters, opt_metric="summary.eirl/reward_correl", debug=True)


def search_meow():
    fixed = dict(
        algo="imeow",
        seed=[0, 42, 100, 532, 3432],
        # hard=[False, True],
        reward_type=["next state", "state", "state-action", "next state only"],
        log_prob_adj_reward=False,
        neg_reward=False,
        maximize_reward=False,
        rew_const=False,  # [True, False],
        training_increments=5,
        # n_expert_demos=10,
        extra_tags=["hp5", "meow"],
        early_learning=False,
        env_name="seals:seals/Swimmer-v1",
        enforce_rew_val_consistency=False,
        gamma=0.999,
        batch_size=64,
        # norm_reward=[False, True],
        # abs_log_probs=[True, False],
        rl_algo="meow",
        consistency_coef=0.,
        lr=1e-3,
        n_eval_episodes=10,
        learner_timesteps=0,
        convex_opt=[True, False],
        calc_log_probs=[True, False],
    )
    bounds = dict(
        q_coef=[1., 10.],
        n_epochs=[10, 250],
        # log_prob_adj_reward=False,
        # neg_reward=False,
        # maximize_reward=False,
        n_expert_demos=[1, 60],
        n_envs=[2, 32],
        # learner_timesteps=[100_000, 3000_000],
        # gamma=[0.98],#[0.8, 0.999],
        # training_increments=[5, 10],
        # lr=[0.00025, 0.0012],
        # l2_weight=[0, 0.002],
        # batch_size=[48, 96],
        # n_envs=[16, 48],
        # enforce_rew_val_consistency=False,
    )
    run_forever(bounds, fixed, run_next_hyperparameters_imeow, opt_metric="summary.IMEow/reward_correl", debug=True)

def search_gflow():
    fixed = dict(
        algo="gflow",
        seed=[0, 42, 100, 532, 3432],
        hard=[False],
        reward_type=["next state only"],#["next state", "state", "state-action", "next state only"],
        training_increments=50,
        extra_tags=["gflow0", "target_back_fixed"],
        early_learning=False,
        env_name="seals:seals/CartPole-v0",
        gamma=0.98,
        batch_size=256,
        # norm_reward=[False, True],
        net_arch=[[256, 256, 256, 256]],
        # learner_timesteps=0,
        n_envs=16,
        flip_cartpole_actions=True,
        use_returns=True,#[True, False],
        use_z=True,
        log_prob_loss=["kl"],#None
        target_log_probs=True,#[False, True],
        target_back_probs=True,
        kl_coef=1.,
        val_coef=0.,
        adv_coef=1.0,
        use_scheduler = False,
        reset_weights = [True, False],
    )
    bounds = dict(
        n_expert_demos=[20, 32],
        learner_timesteps=[1500_000, 3000_000],
        n_epochs=[1500, 3000],
        lr=[1e-8, 1e-4],
    )
    run_forever(bounds, fixed, run_next_hyperparameters, opt_metric="summary.reward/original_ep_return_mean", debug=False)


if __name__ == "__main__":
    search_gflow()
