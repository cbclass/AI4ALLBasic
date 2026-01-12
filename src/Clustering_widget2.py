import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn.cluster import KMeans
from src.utils import read_csv_data
from numpy.random import RandomState


class WidgetClustering2:
    def __init__(self, data=None):
        method = self.applyKMeans
        style = {'description_width': 'initial'}

        self.buttons_ds = []
        for i in range(6):
            button_ds = widgets.ToggleButton(
                value=False,
                description=f'Dataset {i+1}',
                disabled=False,
                button_style='',
                tooltip='Description',
                icon=''
            )
            button_ds.observe(self.on_button_change, names='value')
            self.buttons_ds.append(button_ds)

        self.button_random = widgets.ToggleButton(
            value=False,
            description='Random',
            disabled=False,
            button_style='',
            tooltip='Description',
            icon=''
        )

        self.slider_it = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Iterations",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        self.slider_numclus = widgets.IntSlider(
            value=1,
            min=1,
            max=6,
            step=1,
            description="Number of Clusters",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        self.checkbox_voronoi = widgets.Checkbox(
            value=False,
            description='Show Cluster Areas',
            disabled=False,
            indent=False
        )

        self.button_random.observe(self.on_random_btn_change, names='value')

        
        self.widget = widgets.interactive_output(
            method, 
            {   
                'clusters': self.slider_numclus,
                'iterations': self.slider_it,
                'dataset1': self.buttons_ds[0],
                'dataset2': self.buttons_ds[1],
                'dataset3': self.buttons_ds[2],
                'dataset4': self.buttons_ds[3],
                'dataset5': self.buttons_ds[4],
                'dataset6': self.buttons_ds[5],
                "random_btn": self.button_random,
                "voronoi": self.checkbox_voronoi,
                'X1': widgets.fixed(data)
            }
        )
        
        # Creates Layout
        controls = widgets.VBox([
            widgets.HTML('<div><b>Hyperparameters:</b></div>'),
            self.slider_numclus, 
            self.slider_it, 
            widgets.HTML('<hr style="margin: 1px 0;">'),
            widgets.HTML('<div><b>Datasets:</b></div>'),
            *self.buttons_ds,
            widgets.HTML('<hr style="margin: 1px 0;">'),
            widgets.HTML('<div><b>Initial Cluster Centers:</b></div>'),
            self.button_random,
            widgets.HTML('<hr style="margin: 1px 0;">'),
            widgets.HTML('<div><b>Options:</b></div>'),
            self.checkbox_voronoi
        ])
        
        self.layout = widgets.HBox([controls, self.widget])


    def on_button_change(self, change):
        btns = self.buttons_ds

        for btn in btns:
            if btn != change['owner'] and change['new']:
                btn.value = False

    def on_random_btn_change(self, change):
        self.state = np.random.randint(0, 1000)
        change['owner'].value = False

    def applyKMeans(
            self, 
            clusters, 
            iterations, 
            dataset1, 
            dataset2, 
            dataset3, 
            dataset4, 
            dataset5, 
            dataset6, 
            random_btn,
            voronoi,
            X1):
        
        empty = False
        colors = ['red', "orange", 'green',"blue", "purple", 'brown']
        colors = colors[:clusters]

        if dataset1 and X1 is None:
            data = read_csv_data("data/circular_4.csv") 
            data = data[:, 1:]
        elif dataset2 and X1 is None:
            data = read_csv_data("data/bows.csv") 
            data = data[:, 1:]
        elif dataset3 and X1 is None:
            data = read_csv_data("data/circles_3.csv") 
            data = data[:, 1:]
        elif dataset4 and X1 is None:
            data = read_csv_data("data/lines.csv") 
            data = data[:, 1:]
        elif dataset5 and X1 is None:
            data = read_csv_data("data/circles_4.csv") 
            data = data[:, 1:]
        elif dataset6 and X1 is None:
            data = read_csv_data("data/blocks.csv") 
            data = data[:, 1:]
        else:
            empty = True

        state = self.state if hasattr(self, 'state') else 0

        plt.figure(1 , figsize = (5 , 5) )

        if not empty:
            # Creates a grid of points for coloring the background for the voronoi diagram
            xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 500), np.linspace(0.0, 1.0, 500))
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            if iterations != 0:
                algorithm = KMeans(n_clusters = clusters ,init='random', n_init = 1 ,max_iter=iterations, random_state=state, algorithm='lloyd')

                algorithm.fit(data)
                labels1 = algorithm.labels_
                # creates new array with colors for each point instead of labels
                point_colors = [colors[label] for label in labels1]

                if voronoi:
                    # Prediction for the grid points
                    Z = algorithm.predict(grid_points)
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, colors="grey", alpha=0.1)

                centroids1 = algorithm.cluster_centers_
                plt.scatter(x = data[:,0], y = data[:,1], c = point_colors, s = 1, alpha = 0.5)
                plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 20 , c=colors , alpha = 1.0, edgecolor='black')
            else:
                plt.scatter(x = data[:,0], y = data[:,1], s = 1, alpha = 0.5)

                algorithm = KMeans(n_clusters=clusters, init='random', n_init=1, max_iter=iterations, random_state=state, algorithm='lloyd')

                initial_centers = algorithm._init_centroids(data, x_squared_norms=np.sum(data ** 2, axis=1), init='random', random_state=RandomState(state), sample_weight=np.ones(data.shape[0]))

                plt.scatter(x = initial_centers[: , 0] , y =  initial_centers[: , 1] , s = 20 , c=colors , alpha = 1.0, edgecolor='black')

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.ylabel('x') , plt.xlabel('y')
        plt.show()



widget_instance = WidgetClustering2()
display(widget_instance.layout)