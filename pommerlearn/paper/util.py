import os

import pandas as pd


def get_label(search_mode, model_name, terminal_solver):
    tex_label = ""
    if search_mode == "OnePlayer":
        tex_label += "$\\texttt{SP-MCTS}$"
    elif search_mode == "TwoPlayer":
        tex_label += "$\\texttt{TP-MCTS}$"

    if model_name != "dummy":
        tex_label += f" $\\texttt{{{model_name.upper()}}}$"

    if terminal_solver is not None:
        if terminal_solver:
            tex_label += " $\\texttt{T}$"

    return tex_label


def find_dirs(base_dir, endswith_list):
    endswith_list = endswith_list.copy()
    found_paths = []
    for file in os.listdir(base_dir):
        for i in reversed(range(0, len(endswith_list))):
            endswith = endswith_list[i]
            if file.endswith(endswith):
                found_paths.append(os.path.join(base_dir, file))
                del endswith_list[i]

    if len(endswith_list) > 0:
        print(f"Warning: did not find {endswith_list}.")

    return found_paths


def expand_eval_info(df: pd.DataFrame):
    """
    Splits the information in the eval info column into own columns

    :param df: dataframe
    :returns: new dataframe with added columns
    """
    eval_info_columns = ["count", "mean", "max", "min", "var", "svar"]
    eval_info_attributes = ["eval_time", "eval_nodes", "eval_depth", "eval_seldepth"]
    new_df = df
    new_df[eval_info_attributes] = new_df.EvalInfo.str.split(":", expand=True)
    for attribute in eval_info_attributes:
        new_attribute_col_vals = new_df[attribute].str.split(",", expand=True).astype(float)
        new_df[[f"{attribute}_{col}" for col in eval_info_columns]] = new_attribute_col_vals
    return new_df


def add_wld_columns(df: pd.DataFrame, agent=0):
    """
    Adds proportional win/loss/draw columns for an agent.

    :param df: dataframe
    :param agent: agent id
    :returns: new dataframe with added win/loss/draw columns
    """
    new_df = df
    new_df[f"RelativeWin{agent}"] = df[f"Wins{agent}"] / df.Episodes
    new_df[f"RelativeLoss{agent}"] = (df.Episodes - df[f"Wins{agent}"] - df.Draws - df.NotDone) / df.Episodes
    new_df[f"RelativeDraw{agent}"] = (df.Draws + df.NotDone) / df.Episodes
    return new_df
