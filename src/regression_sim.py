import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import random as rnd

def generateData(n=10,xmin=0,xmax=1,bmin=-2,bmax=2,mmin=-2,mmax=2):
    m=rnd.uniform(mmin,mmax)
    b=rnd.uniform(bmin,bmax)

    x=[rnd.uniform(xmin,xmax) for i in range(n)]

    y=[m*x+b for x in x]

    sigma=(max(y)-min(y))*0.2
    y=[y + rnd.gauss(sigma=sigma) for y in y]
    
    data=np.array([x,y]).transpose()

    return data
    
class WidgetRegression:
    def __init__(self,data):
        self.colors=['dimgray','silver','darksalmon','tan','darkseagreen',
                'mediumaquamarine','cadetblue','lightsteelblue','thistle']
        method = self.plotData
        style = {'description_width': 'initial'}

        self.slider_m = widgets.FloatSlider(
            value=0,
            min=-4,
            max=4,
            step=0.01,
            description="m",
            tooltip='Steigung',
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=style
        )

        self.value_m=widgets.BoundedFloatText(
            value=0,
            min=-4,
            max=4,
            step=0.1,
            description='m:',
            tooltip='Steigung',
            disabled=False
            #position='f'
        )

        mValues = widgets.link((self.slider_m, 'value'), (self.value_m, 'value'))
        
        self.slider_b = widgets.FloatSlider(
            value=0,
            min=-4,
            max=4,
            step=0.01,
            description="b",
            tooltip='y-Achsenabschnitt',
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=style
        )

        self.value_b=widgets.BoundedFloatText(
            value=0,
            min=-4,
            max=4,
            step=0.1,
            description='b:',
            tooltip='y-Achsenabschnitt',
            disabled=False
            #position='f'
        )

        bValues = widgets.link((self.slider_b, 'value'), (self.value_b, 'value'))

        self.anzeigeErrors=widgets.Checkbox(
            value=False,
            description='Anzeige der Risiduen',
            disabled=False,
            indent=False
        )
        
        self.text_block = widgets.HTML(value="<b>Regressionsgleichung:</b> y =" + str(self.slider_b.value))

        self.widget = widgets.interactive_output(method, 
                 {'m': self.slider_m,
                  'b': self.slider_b,
                  'residuals':self.anzeigeErrors,
                  'data': widgets.fixed(data)})
        
        # Layout erstellen
        controls = widgets.VBox([self.text_block, self.slider_m, self.value_m, self.slider_b, self.value_b, 
                                 self.anzeigeErrors])
                            
        self.layout = widgets.HBox([controls, self.widget])
        
        
    def plotData(self, m, b, residuals,data):
        plt.figure(figsize=(5, 4), dpi=100)

        coef = m
        intercept = b
        sum_squared_error = 0.0

        plt.scatter(data[:, 0], data[:, 1], color='black', label='Datenpunkte', s=3.0)
        y_pred = coef * data[:, 0] + intercept
        y0=intercept
        y1=coef+intercept
        
        for i in range(len(data[:, 0])):
            y_p = coef * data[i, 0] + intercept
            # Summe der Fehlerquadrate berechnen
            sum_squared_error += (data[i, 1] - y_p) ** 2
            if residuals:
                # Darstellung der Residuen
                #Bestimmung der Farbe
                
                c=self.colors[i%len(self.colors)]

                # Zeichnen der Fehlerlinie
                plt.plot([data[i, 0],data[i, 0]], [data[i, 1], y_pred[i]], color=c, linestyle='--', linewidth=0.5)

                # Berechnen der Länge der Linie / des Fehlers
                line_length = abs(data[i, 1] - y_p)
                
                # Anzeigen des Fehlers
                plt.text(data[i, 0]+0.01, (data[i, 1]+y_p)/2, f'{line_length:.2f}', fontsize=6, color=c)




        self.text_block.value = ("<b>Regressionsgleichung:</b> <br> y = {:.2f} {} {:.2f} · x".format( b, '+' if m >= 0 else '-', abs(m)) + \
                                     "<br><b>Summe quadratischer Fehler:</b> <br>{:.4f}".format(sum_squared_error)) + \
                                     "<br><b>Parameter:</b>"
            
        plt.plot([0,1], [y0,y1], color='blue', linewidth=1, label='Gerade')

        ymin=min(data[:,1].min(),min(y0,y1))-0.1
        ymax=max(data[:,1].max(),max(y0,y1))+0.1
        plt.xlim(0, 1)
        plt.ylim(ymin, ymax)
        plt.show()



