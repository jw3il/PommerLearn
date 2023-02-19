import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.util import get_label, expand_eval_info, add_wld_columns

from matplotlib_settings import set_matplotlib_font_size, init_plt

df = pd.read_csv("20230216_222725_cpp_eval.csv")

print(df.columns)

df = expand_eval_info(df)
df = add_wld_columns(df)
df["EpisodeSteps"] = df.TotalSteps / df.Episodes

# aggregate over all runs
df_grouped = df.groupby(["OpponentModel", "SearchMode", "ModelName"])
df_mean = df_grouped.mean().reset_index()
df_std = df_grouped.std().reset_index()
df_count = df_grouped.count().reset_index()

print(df_mean.columns)
print(df_mean.OpponentModel.unique())
print(df_mean.SearchMode.unique())


def eval_stat_string(mean_df, attribute_str):
    assert len(mean_df) == 1
    return (
        f"{mean_df[attribute_str + '_mean'].item():.2f} +/- {mean_df[attribute_str + '_var'].map(lambda x: np.sqrt(x)).item():.2f}"
        f" (min: {mean_df[attribute_str + '_min'].item():.2f}, max: {mean_df[attribute_str + '_max'].item():.2f})"
    )

def print_run_info(df_filter):
    assert len(df_mean[df_filter]) == 1
    print(
        f"{df_count[df_filter].OpponentModel.item()}, {df_count[df_filter].SearchMode.item()}, {df_count[df_filter].ModelName.item()}"
        f" ({df_count[df_filter].RelativeWin0.item()} results)"
    )
    depth_factor = 1.0 if df_count[df_filter].SearchMode.item() == "OnePlayer" else 0.5
    print(
        "Win/Draw/Depth/Time/Steps for LaTeX table: \n"
        f"${df_mean[df_filter]['RelativeWin0'].item():.2f} \pm {df_std[df_filter]['RelativeWin0'].item():.2f}$"
        " & "
        f"${df_mean[df_filter]['RelativeDraw0'].item():.2f} \pm {df_std[df_filter]['RelativeDraw0'].item():.2f}$"
        " & "
        f"${depth_factor * df_mean[df_filter]['eval_depth_mean'].item():.2f} \pm {df_mean[df_filter]['eval_depth_var'].map(lambda x: depth_factor * np.sqrt(x)).item():.2f}$"
        " & "
        f"${df_mean[df_filter]['eval_time_mean'].item():.2f} \pm {df_mean[df_filter]['eval_time_var'].map(lambda x: np.sqrt(x)).item():.2f}$"
        " & "
        f"${df_mean[df_filter]['EpisodeSteps'].item():.2f} \pm {df_std[df_filter]['EpisodeSteps'].item():.2f}$"
    )
    print(
        f"Eval nodes: {eval_stat_string(df_mean[df_filter], 'eval_nodes')} \n"
        f"Eval time: {eval_stat_string(df_mean[df_filter], 'eval_time')} \n"
        f"Eval depth: {eval_stat_string(df_mean[df_filter], 'eval_depth')} \n"
        f"Eval seldepth: {eval_stat_string(df_mean[df_filter], 'eval_seldepth')}"
    )


print("SL")
for search_mode in ["OnePlayer", "TwoPlayer"]:
    for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
        df_filter = (df_mean.OpponentModel == opponent_model) & (df_mean.SearchMode == search_mode) & (df_mean.ModelName == "sl")
        print_run_info(df_filter)
        print()

print("SL2RL")
for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
    for search_mode in ["OnePlayer"]:
        df_filter = (df_mean.OpponentModel == opponent_model) & (df_mean.SearchMode == search_mode) & (df_mean.ModelName == "sl2rl")
        print_run_info(df_filter)
        print()

print("RL")
for opponent_model in ["SimpleUnbiasedAgent", "RawNetAgent"]:
    for search_mode in ["OnePlayer"]:
        df_filter = (df_mean.OpponentModel == opponent_model) & (df_mean.SearchMode == search_mode) & (df_mean.ModelName == "rl")
        print_run_info(df_filter)
        print()