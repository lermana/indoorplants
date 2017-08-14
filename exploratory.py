import matplotlib.pyplot as plt
import scipy.stats as sp

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


def _get_hist(series, bins=None, **kwargs):
    """helper function that handles both categorical and 
    numeric data"""
    if series.dtype in (int, float):
        if bins is None:
            bins = min(int(round(series.nunique(), -2) / 10), 100)
        return series.hist(bins=bins, **kwargs)
    else:
        return series.value_counts().sort_index(
                        ).plot.bar(**kwargs)


def feature_hist_by_class(table, class_col, feature, bins=None, 
                          figsize=(11, 8), **kwargs):
    """plot histogram of feature, with data color-coded by class"""
    classes = table[class_col].unique()
    ax = _get_hist(table.loc[table[class_col] == classes[0], 
                             feature], 
                   bins=bins, label=str(classes[0]), 
                   alpha=.5, figsize=figsize, **kwargs)

    for i, val in enumerate(classes[1:], 1):
        _get_hist(table.loc[table[class_col] == val, feature],
                  ax=ax, alpha=.5, label=str(val), bins=bins,
                  color='C{}'.format(i), **kwargs)

    plt.legend(loc='best')
    title = plt.title('{} histogram, across {}'.format(class_col, 
                                                       feature))


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


def scatter_by_class(table, class_col, x, y, figsize=(11, 8)):
    """scatter two features, color code by class"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    classes = table[class_col].unique()
    filtered = table.loc[table[class_col] == classes[0], :]
    ax.scatter(x=filtered[x], y=filtered[y], alpha=.5, label=classes[0])

    for i, val in enumerate(classes[1:], 1):
        filtered = table.loc[table[class_col] == val, :]
        ax.scatter(x=filtered[x], y=filtered[y], alpha=.5, label=val,
                   color='C{}'.format(i))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend(loc='best')
    title = plt.title('{}: {} vs. {}'.format(class_col, y, x))


def center_scale_plot(series, center_func, scale_func, bins=None,
                      figsize=(11, 8)):
    """produce histogram overlayed with bands corresponding to
    units of scale from the center. user supplies center and 
    scale functions"""
    
    # create figure, plot hist, and get max dimensions
    plt.figure()
    if bins is None:
        bins = min(int(round(series.nunique(), -2) / 10), 100)

    ax = series.hist(figsize=figsize, bins=bins)

    ymaxtick = ax.get_yticks()[-1]
    xmaxtick = ax.get_xticks()[-1]
    
    # get center and scale
    center = round(center_func(series), 1)
    scale = round(scale_func(series), 1)
    
    # plot box with center & scale
    bbox = {'fc': '1', 'pad': 3}
    ax.annotate(
    """{}: {}
{}: {}""".format(center_func.__name__,
                     round(center, 1),
                     scale_func.__name__,
                     round(scale, 1)),
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
        bandn1 = round(center - scale, 1)
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
    band1 = round(center + scale, 1)
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
        bandn2 = round(center - 2 * scale, 1)
        ax.axvspan(bandn2,
                   bandn1, 
                   alpha=0.5, 
                   color='y')

    # plot second band
    band2 = round(center + 2 * scale, 1)
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
        bandn3 = round(center - 3 * scale, 1)
        ax.axvspan(bandn3,
                   bandn2, 
                   alpha=0.5, 
                   color='purple')

    # plot third band
    band3 = round(center + 3 * scale, 1)
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