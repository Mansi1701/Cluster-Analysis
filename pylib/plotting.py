import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

   
def contour_plot(C, betas, X, y):
    
    # define axis limits
    mx = 15            # maximum value on x beta axis
    hx = 1             # step size on x axis in the mesh
    sx = 2*int(mx/hx)  # number of grid units on x axis

    my = 4             # maximum value on y beta axis
    hy = 0.2           # step size on y axis in the mesh
    sy = 2*int(my/hy)  # number of grid units on y axis

    beta_colors = ['#ba2121ff', '#42a5f5ff', '#efa016ff']

    # prepare for the grid plot 
    xx, yy = np.meshgrid(np.arange(-mx, mx, hx),
                         np.arange(-my, my, hy))

    zz = np.apply_along_axis(C, 1, np.c_[xx.ravel(), yy.ravel()]).reshape(sy, sx)

    # make the figure and subplots
    fig = plt.figure(figsize=(15, 7))
    gridspec.GridSpec(3, 3)

    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.contour(xx, yy, zz, 50, colors='k')
    plt.xlabel(r'$\beta_0$')
    plt.ylabel(r'$\beta_1$');
    for idx in range(3):
        plt.plot(betas[idx][0], betas[idx][1], marker='o',
                 markersize=8, color=beta_colors[idx])  

    for idx in range(3):
        plt.subplot2grid((3, 3), (idx, 2))
        plt.plot(X, y, 'o', color='k')
        plt.plot(X, betas[idx][1]*X + betas[idx][0], color=beta_colors[idx]);
        plt.ylabel('Electoral votes');

    plt.xlabel('State population in millions')

    fig.tight_layout()
    
def plot_grad_desc(C, betas):
    
    def plotter(num_steps):
    
        # define axis limits
        maxx = 6                    # maximum value on x beta axis
        minx = -2                   # minimum value on x beta axis
        hx = 0.1                    # step size on x axis in the mesh
        sx = int((maxx - minx)/hx)  # number of grid units on x axis

        maxy = 2.5                  # maximum value on y beta axis
        miny = 0.5                  # minimum value on y beta axis
        hy = 0.05                   # step size on y axis in the mesh
        sy = int((maxy - miny)/hy)  # number of grid units on y axis

        # prepare for the grid plot 
        xx, yy = np.meshgrid(np.arange(minx, maxx, hx),
                             np.arange(miny, maxy, hy))

        zz = np.apply_along_axis(C, 1, np.c_[xx.ravel(), yy.ravel()]).reshape(sy, sx)

        plt.contour(xx, yy, zz, 50, colors='k')
        plt.plot(betas[:num_steps+1, 0],
                 betas[:num_steps+1, 1], 
                 marker='o', markersize=5,
                c='#42a5f5ff')
        plt.xlabel(r'$\beta_0$')
        plt.ylabel(r'$\beta_1$')
        
    return plotter
    
    
def telco_plot(df):
    
    color = '#42a5f5ff'

    def plotter(column):       
        if not isinstance(df[column][0], (np.integer, np.float)):
            plot_values = {}
            for name, group in df.groupby(column):
                churned = group[group['churn']]
                percentage_churned = churned.shape[0]/group.shape[0]
                plot_values[name] = percentage_churned
            plt.bar(range(len(plot_values)), list(plot_values.values()),
                    align='center', color=color)
            plt.xticks(range(len(plot_values)), list(plot_values.keys()), 
                       rotation=(90 if (len(plot_values) > 3) else 0),
                      fontsize=(6 if (len(plot_values) > 3) else 12))
            plt.xlabel(column)
            plt.ylabel('Percentage churned')
        else:
            churned = df[df['churn']]
            not_churned = df[~df['churn']] 
            vp = plt.violinplot([not_churned[column].values, 
                            churned[column].values], 
                           [0, 1], showmeans=True) 
            for pc in vp['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
            plt.xticks([0, 1], ['not churned', 'churned'])
            plt.ylabel(column)
    
    return plotter

def predicted_probabilities_heatmap(model, columns=None):
    
    # prepare the 2-d feature space
    x_min = -1000
    x_max = 45000
    mesh = np.column_stack([a.reshape(-1) for a in np.meshgrid(np.r_[x_min:x_max:100j], np.r_[x_min:x_max:100j])])
    if columns is not None:
        mesh = pd.DataFrame(mesh, columns=columns)

    # compute predicted probabilities
    zz = model.predict_proba(mesh).T[1]

    # plot predicted probabilities
    plt.imshow(zz.reshape(100,100), cmap=plt.cm.RdBu_r, origin='lower',
               extent=(x_min, x_max, x_min, x_max), vmin=0, vmax=1)
    cb = plt.colorbar()
    cb.set_label('probability of Hotel/Restaurant/Cafe')
    
    
def plot_varying_threshold(df_wholesale, model, feature_1, feature_2, color_dict):   
    
    labels = {1: ['true positives', 'false negatives'], 0: ['false positives', 'true negatives']}
    X = df_wholesale[[feature_1, feature_2]]
    y = df_wholesale['Channel']
    df_wholesale['model_probs'] = model.predict_proba(X).T[1]
    x = np.linspace(X[feature_1].min(), X[feature_1].max(), 100)    
    
    def plotter(threshold=0.5):
        
        # compute the linear boundary
        coef_1 = model.coef_[0][0]
        coef_2 = model.coef_[0][1]
        intercept = model.intercept_
        boundary = (-coef_1*x - intercept - np.log(1/threshold -1))/coef_2
        
        # plot boundary
        plt.plot(x, boundary, 'k-', label='linear boundary', lw=1)

        # plot the data
        groups = df_wholesale.groupby('Channel')
        for name, group in groups:
            hotels = group[group['model_probs']>threshold]
            plt.scatter(hotels[feature_1], hotels[feature_2], edgecolor=color_dict[1],
                        lw=1, s=25, c=color_dict[name], label=labels[name][0])
            
            not_hotels = group[group['model_probs']<threshold]
            plt.scatter(not_hotels[feature_1], not_hotels[feature_2], edgecolor=color_dict[0], 
                        lw=1, s=25, c=color_dict[name], label=labels[name][1])

        plt.ylim([-2000, 40000])
        plt.title('Wholesale customer spending')
        plt.xlabel('Annual spending on {} products [euros]'.format(feature_1.lower()))
        plt.ylabel('Annual spending on {} products [euros]'.format(feature_2.lower()))
        plt.legend(loc='upper right')
        plt.grid(False)
        
        # compute and display precision and recall
        y_pred = (df_wholesale['model_probs'] > threshold).astype(int)
        precision = metrics.precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)
        plt.title('Precision: %0.2f, Recall: %0.2f' % (precision, recall))
           
    return plotter

def plot_kmeans(X, n=3, start=None):
    colors = ['#ba2121ff', '#42a5f5ff', '#efa016ff', '#000000ff', '#ffffffff', '#6f7c91ff']

    if start is not None:
        n = start.shape[0]
    else:
        start = X[::X.shape[0]//n,:]
    
    def func(train_steps=0):
        
        if train_steps:
            km = KMeans(n_clusters=n, max_iter=train_steps, n_init=1, init=start)
            km.fit(X)
            centers = km.cluster_centers_.T
            cols = [colors[label] for label in km.labels_]
        else:
            centers = start.T
            cols = '0.5'
        plt.scatter(*X.T, s=20, c=cols)
        plt.scatter(*centers, c=colors[:len(centers.T)], marker='*', s=150,
                    linewidth=1, edgecolors='k')
        plt.axis('image')
        plt.title('Toy Example Data with Only Two Features')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    return func
