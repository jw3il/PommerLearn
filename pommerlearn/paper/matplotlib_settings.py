import matplotlib

# use tex
matplotlib.rcParams['text.usetex'] = True

# export with higher dpi
matplotlib.rcParams['savefig.dpi'] = 300

# switch to True for high res 3d surface plots (will take longer)
SURFACE_PLOT_HIGH_TRI_COUNT = True


def set_matplotlib_font_size(small_size=8, medium_size=10, bigger_size=12):
    """
    Sets matplotlib font sizes, see
    https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot.

    :param small_size: size for font, axes title, xtick, ytick, legend
    :param medium_size: size for axes label
    :param bigger_size: size for figure title
    """
    import matplotlib.pyplot as plt
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title


def init_plt():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.grid.axis'] = 'y'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['legend.frameon'] = 'True'
    plt.rcParams['legend.framealpha'] = '1.0'
    plt.rc('font', **{'family': 'serif'})
    # http://phyletica.org/matplotlib-fonts/
    # not necessary as we use tex
    # plt.rcParams['pdf.fonttype'] = 42
    # plt.rcParams['ps.fonttype'] = 42

