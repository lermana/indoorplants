import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

from . import features

def qq_plot(series, figsize=(11, 8)):
    """compare given series to Normal distirbution
    with matching location & scale"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    arrs, res = sp.probplot(series, 
                            dist=sp.norm, 
                            sparams=(series.mean(), series.std()), 
                            plot=ax)
    bbox = {'fc': '1', 'pad': 3}
    xy = (ax.get_xticks()[-2], ax.get_yticks()[2])
    text = ax.annotate(r'$R^2$: {}'.format(round(res[2], 3)),
                     xy=xy,
                     bbox=bbox)
    title = ax.set_title('Q-Q plot')
    return ax


def feature_value_counts_by_class(table, class_col, feature, 
                                  figsize=(11, 8), **kwargs):
    """plot histogram of feature, with data color-coded by class"""
    ax = table.groupby([class_col, feature]
             ).size(
             ).unstack(0
             ).plot.bar(stacked=True, figsize=figsize, 
                        alpha=.5, **kwargs)
    
    plt.legend(loc='best')
    title = plt.title('{} histogram, across {}'.format(class_col, feature))
    return ax


def feature_hist_by_class(eda_df, class_col, feature, bins=None): 
    """
    Plots a histogram of passed feature, broken out by class, which
    must be of a binary nature.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax = eda_df[eda_df[class_col] == 1
               ][feature].hist(ax=ax, bins=bins, color="orange", alpha=.5)
    ax = eda_df[eda_df[class_col] == 0
               ][feature].hist(ax=ax, bins=bins, color="blue", alpha=.5)
    l = ax.set_ylabel("count")
    l = ax.set_xlabel(f"{feature}")
    t = ax.set_title(f"hist: {feature}, by {class_col} (orange=1, blue=0)")
    return ax


def classes_across_feature(table, class_col, feature, figsize=(11, 8)):
    """scatter feature against class"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    classes = table[class_col].unique()
    filtered = table.loc[table[class_col] == classes[0], :]
    ax.scatter(x=filtered[feature], y=filtered[class_col], label=classes[0])

    for i, val in enumerate(classes[1:], 1):
        filtered = table.loc[table[class_col] == val, :]
        ax.scatter(x=filtered[feature],y=filtered[class_col],
                  label=val, color='C{}'.format(i))

    plt.yticks([classes[0], classes[-1]])
    plt.legend(loc='best')
    title = plt.title('{}: by {}'.format(class_col, feature))
    return ax


def scatter_by_class(table, class_col, x, y, figsize=(11, 8), alpha=.5):
    """scatter two features, color code by class"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    classes = table[class_col].unique()
    filtered = table.loc[table[class_col] == classes[0], :]
    ax.scatter(x=filtered[x], y=filtered[y], alpha=alpha, label=classes[0])

    for i, val in enumerate(classes[1:], 1):
        filtered = table.loc[table[class_col] == val, :]
        ax.scatter(x=filtered[x], y=filtered[y], alpha=alpha, label=val,
                   color='C{}'.format(i))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend(loc='best')
    title = plt.title('{}: {} vs. {}'.format(class_col, y, x))
    return ax


def box_plot_by_class(eda_df, class_col, feature):
    
    ax = eda_df[[f"{feature}", f"{class_col}"]
              ].pivot(values=f"{feature}", columns=f"{class_col}"
              ).boxplot(figsize=(11, 8))
    l= ax.set_xlabel(f"{class_col}")
    l= ax.set_ylabel(f"{feature}")
    t = ax.set_title(f"box plot: {feature} by {class_col}")
    return ax


def center_scale_plot(series, center_func=np.mean, scale_func=np.std, bins=None,
                      figsize=(11, 8), return_bins=False, ndigits=4):
    """produce histogram overlayed with bands corresponding to
    units of scale from the center. user supplies center and 
    scale functions"""
    
    # create figure, plot hist, and get max dimensions
    if series.dtype not in (int, float):
        raise TypeError("Must pass numeric data for histogram!")

    plt.figure()
    if bins is None:
        bins = max(min(int(round(series.nunique(), -2) / 10), 100), 10)

    ax = series.hist(figsize=figsize, bins=bins)

    ymaxtick = ax.get_yticks()[-1]
    xmaxtick = ax.get_xticks()[-1]
    
    # get center and scale
    center = round(center_func(series), ndigits)
    scale = round(scale_func(series), ndigits)
    
    # plot box with center & scale
    bbox = {'fc': '1', 'pad': 3}
    ax.annotate(
    """{}: {}
{}: {}""".format(center_func.__name__,
                     round(center, ndigits),
                     scale_func.__name__,
                     round(scale, ndigits)),
        xy=(.8* xmaxtick, .8 * ymaxtick),
        bbox=bbox)

    # annotate center
    ax.annotate("{}".format(center), 
            xy=(center, 
                .8 * ymaxtick),
            xytext=(1.19 * center, 
                    .8 * ymaxtick),
            arrowprops={'facecolor': 'black', 
                        'arrowstyle': "-|>"})
    
    # plot first negative band if applicable
    if ax.get_xticks()[0] < center - scale:
        bandn1 = round(center - scale, ndigits)
        ax.axvspan(bandn1,
                   center, 
                   alpha=0.5, 
                   color='g')
        ax.axvline(center, c='b')
    else:
        ax.axvspan(ax.get_xticks()[0],
                   center, 
                   alpha=0.2, 
                   color='blue')

    # plot first band
    band1 = round(center + scale, ndigits)
    ax.axvspan(center,
               band1, 
               alpha=0.5, 
               color='g')    

    # annotate band 1
    ax.annotate("{}".format(band1), 
            xy=(band1, 
                .7 * ymaxtick),
            xytext=(band1 + .19 * center, 
                    .7 * ymaxtick),
            arrowprops={'facecolor': 'black', 
                        'arrowstyle': "-|>"})

    # plot negative second band if applicable
    if ax.get_xticks()[0] < center -  2 * scale:
        bandn2 = round(center - 2 * scale, ndigits)
        ax.axvspan(bandn2,
                   bandn1, 
                   alpha=0.5, 
                   color='y')

    # plot second band
    band2 = round(center + 2 * scale, ndigits)
    ax.axvspan(band1,
               band2, 
               alpha=0.5, 
               color='y')

    # annotate second band
    ax.annotate("{}".format(band2), 
            xy=(band2, 
                .6 * ymaxtick),
            xytext=(band2 + .19 * center, 
                    .6 * ymaxtick),
            arrowprops={'facecolor': 'black', 
                        'arrowstyle': "-|>"})
    
    # plot negative third band if applicable
    if ax.get_xticks()[0] < center -  3 * scale:
        bandn3 = round(center - 3 * scale, ndigits)
        ax.axvspan(bandn3,
                   bandn2, 
                   alpha=0.5, 
                   color='purple')

    # plot third band
    band3 = round(center + 3 * scale, ndigits)
    ax.axvspan(band2,
               band3, 
               alpha=0.5, 
               color='purple')
    
    # annotate third band
    ax.annotate("{}".format(band3), 
            xy=(band3, 
                .5 * ymaxtick),
            xytext=(band3 + .19 * center, 
                    .5 * ymaxtick),
            arrowprops={'facecolor': 'black', 
                        'arrowstyle': "-|>"})

    ax.title.set_text('{}: histogram, {} & {}'.format(
                            series.name, 
                            center_func.__name__, 
                            scale_func.__name__))
    if return_bins:
        return ax, bins
    else:
        return ax


def center_scale_with_peak_trough(series, plot_troughs=True, plot_peaks=False,
                                  bins=100, center_scale_plot_kwargs=None, 
                                  trough_plt_kwargs=None, peak_plt_kwargs=None):
    """
    Produces `center_scale_plot` but with the option to add peak and / or trough
    lines to the resulting histogram. See `features.find_frequency_peaks_and_troughs` 
    for more information on peak and trough calculation.
    """
    def plot_optima(optima, **kwargs):
        for opt in np.mean(optima, axis=1):
            
            if "linestyle" not in kwargs:
                kwargs["linestyle"] = "--"
            
            if "linewidth" not in kwargs:
                kwargs["linewidth"] = 1

            ax.axvline(opt, 0, y_max, **kwargs)
    
    if center_scale_plot_kwargs is None:
        center_scale_plot_kwargs = {"center_func": np.mean,
                                    "scale_func": np.std}
    
    if trough_plt_kwargs is None:
        trough_plt_kwargs = {}
        
    if peak_plt_kwargs is None:
        peak_plt_kwargs = {}
    
    ax = center_scale_plot(series, bins=bins, **center_scale_plot_kwargs)
    peaks, troughs = features.find_frequency_peaks_and_troughs(series, bins=bins)
    
    y_max = ax.get_ybound()[1]
    
    if plot_troughs:
        plot_optima(troughs, **trough_plt_kwargs)
    
    if plot_peaks:
        plot_optima(peaks, **peak_plt_kwargs)
    
    return ax