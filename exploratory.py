import matplotlib.pyplot as plt

def classs_across_feature(table, class_col, feature):
    classes = table[class_col].unique()

    ax = table[table[class_col] == classes[0]
              ].plot.scatter(figsize=(11, 8),
                        x=feature,
                        y=class_col,
                        label=classes[0])

    for i, val in enumerate(classes[1:], 1):
        table[table[class_col] == val
                  ].plot.scatter(ax=ax,
                                  x=feature,
                                  y=class_col,
                                  label=val,
                                  color='C{}'.format(i))
    plt.yticks([classes[0], classes[-1]])
    plt.legend(loc='best')
    title = plt.title('{}: by {}'.format(class_col, feature))



def center_scale_plot(series, center_func, scale_func):
    # create figure and get max dimensions
    plt.figure()
    num_bins = min(int(round(series.nunique(), -2) / 10), 100)
    ax = series.hist(figsize=(11, 8), bins=num_bins)
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