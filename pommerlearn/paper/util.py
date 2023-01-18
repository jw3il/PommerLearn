def get_label(search_mode, model_name, terminal_solver):
    tex_label = ""
    if search_mode == "OnePlayer":
        tex_label += "$\\texttt{MCTS}_1$"
    elif search_mode == "TwoPlayer":
        tex_label += "$\\texttt{MCTS}_2$"

    if model_name != "dummy":
        tex_label += f" $\\texttt{{{model_name.upper()}}}$"

    if terminal_solver is not None:
        if terminal_solver:
            tex_label += " $\\texttt{T}$"

    return tex_label