import os


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
