import pandas as pd


def model_type_mapping(model_str: str):
    if model_str.startswith("model-sl-"):
        return "sl"
    elif model_str.startswith("model-sl2rl-"):
        return "sl2rl"
    elif model_str.startswith("model-rl-"):
        return "rl"
    else:
        raise ValueError(f"Unknown model type in model {model_str}")


MODEL_TYPE = "ModelName"
SEARCH_MODE = "1v1"
OPPONENT_MODEL = "planning"
df = pd.read_csv("20230220_py_eval_100.csv")

df[MODEL_TYPE] = df.model.map(model_type_mapping)
df[f"relative_a1"] = df["a1"] / 100
df[f"relative_ties"] = df["ties"] / 100

df_grouped = df.groupby([MODEL_TYPE, SEARCH_MODE, OPPONENT_MODEL])
df_mean = df_grouped.mean().reset_index()
df_std = df_grouped.std().reset_index()
df_count = df_grouped.count().reset_index()


def print_stats(df_filter):
    mean = df_mean[df_filter]
    std = df_std[df_filter]
    count = df_count[df_filter]

    assert len(mean) == 1

    print(
        f"{mean[MODEL_TYPE].item()} {mean[OPPONENT_MODEL].item()} {mean[SEARCH_MODE].item()}"
        f" ({count['relative_a1'].item()} entries)"
    )
    print(
        f"For LaTeX: ${mean['relative_a1'].item():.2f} \pm {std['relative_a1'].item():.2f}$ & ${mean['relative_ties'].item():.2f} \pm {std['relative_ties'].item():.2f}$"
    )
    print(
        "Additional info:"
        f" timeout percentage {mean['timeouts'].item() / mean['ties'].item() * 100:.2f}"
        f", steps {mean['steps_mean'].item():.2f} +/- {mean['steps_std'].item():.2f}"
    )


print("SL")
for search_mode_two_player in [False, True]:
    for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
        print_stats((df_mean[MODEL_TYPE] == "sl") & (df_mean[OPPONENT_MODEL] == opponent_model) & (df_mean[SEARCH_MODE] == search_mode_two_player))

print()
print("SL2RL")
for search_mode_two_player in [False]:
    for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
        print_stats((df_mean[MODEL_TYPE] == "sl2rl") & (df_mean[OPPONENT_MODEL] == opponent_model) & (df_mean[SEARCH_MODE] == search_mode_two_player))

print()
print("RL")
for search_mode_two_player in [False]:
    for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
        print_stats((df_mean[MODEL_TYPE] == "rl") & (df_mean[OPPONENT_MODEL] == opponent_model) & (df_mean[SEARCH_MODE] == search_mode_two_player))



