import streamlit as st
import numpy as np
import pandas as pd
import time 
import altair as alt
import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from collections import deque
from math import log

class plot_trajectory: 
    def __init__(self, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa):
        self.nt = nt
        self.ntrans = ntrans
        self.nc = nc
        self.dt = dt
        self.massa = massa
        self.gamma = gamma
        self.ctelastica = ctelastica
        self.alfa = alfa
        self.dalfa = dalfa
        self.xa = xa
        self.dxa = dxa

    def openframe(self):
        left  = 0.125  # the left side of the subplots of the figure
        right = 0.9    # the right side of the subplots of the figure
        bottom = 0.4   # the bottom of the subplots of the figure
        top = 1.2      # the top of the subplots of the figure
        wspace = 0.4   # the amount of width reserved for blank space between subplots
        hspace = 0.4  

        self.fig, self.axs = plt.subplots(2, 2)
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        self.the_plot = st.pyplot(plt)

    def xxplotly(self):
        fig = go.Figure(
            data=[go.Scatter(x = self.t, y = self.x,
                        name="frame",
                        mode="lines",
                        line=dict(width=2, color="blue"))#,
                # go.Scatter(x=x, y=y,
                #       name="curve",
                #       mode="lines",
                #       line=dict(width=2, color="blue"))
                ],
            layout=go.Layout(width=800, height=300,
                        xaxis=dict(range=[min(self.t), max(self.t)], autorange=False, zeroline=False),
                        yaxis=dict(range=[min(self.x), max(self.x)], autorange=False, zeroline=False),
                        title="x(t)",
                        hovermode="closest",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),

            frames=[go.Frame(
                data=[go.Scatter(
                    x=self.t[:k],
                    y=self.x[:k],
                    mode="lines",
                    line=dict(color="red", width=2))
                ]) for k in range(self.nt)]
        )
        st.write(fig)
        #fig.show()

    def vvplotly(self):
        fig = go.Figure(
            data=[go.Scatter(x = [0], y = [0],
                        name="frame",
                        mode="lines",
                        line=dict(width=2, color="blue"))
                ],
            layout=go.Layout(width=800, height=300,
                        xaxis = dict(range=[min(self.t), max(self.t)], autorange=False, zeroline=False),
                        yaxis = dict(range=[min(self.v), max(self.v)], autorange=False, zeroline=False),
                        title = "v(t)",
                        hovermode="closest",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),
            frames=[go.Frame(
                data=[go.Scatter(
                    x=self.t[:k],
                    y=self.v[:k],
                    mode="lines",
                    line=dict(color="red", width=2))
                ]) for k in range(self.nt)]
        )
        st.write(fig)

    def zetaplotly(self):
        fig = go.Figure(
            data=[go.Scatter(x = [0], y = [0],
                        name="frame",
                        mode="lines",
                        line=dict(width=2, color="blue"))
                ],
            layout=go.Layout(width=800, height=300,
                        xaxis = dict(range=[min(self.t), max(self.t)], autorange=False, zeroline=False),
                        yaxis = dict(range=[min(self.zeta), max(self.zeta)], autorange=False, zeroline=False),
                        title = "\zeta(t)",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),
            frames=[go.Frame(
                data=[go.Scatter(
                    x=self.t[:k],
                    y=self.zeta[:k],
                    mode="lines",
                    line=dict(color="red", width=2))
                ]) for k in range(self.nt)]
        )
        #fig.show()
        st.write(fig)

    def eeplotly(self):
        fig = go.Figure(
            data=[go.Scatter(x = [0], y = [0],
                        name="frame",
                        mode="lines",
                        line=dict(width=2, color="blue"))
                ],
            layout=go.Layout(width=800, height=300,
                        xaxis = dict(range=[min(self.t), max(self.t)], autorange=False, zeroline=False),
                        yaxis = dict(range=[min(self.e), max(self.e)], autorange=False, zeroline=False),
                        title = "E(t)",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),
            frames=[go.Frame(
                data=[go.Scatter(
                    x=self.t[:k],
                    y=self.e[:k],
                    mode="lines",
                    line=dict(color="red", width=2))
                ]) for k in range(self.nt)]
        )
        #fig.show()
        st.write(fig)
        
    def simulate(self):

        xx = np.zeros((int(self.nt-self.ntrans)))
        vv = np.zeros((int(self.nt-self.ntrans)))
        ee = np.zeros((int(self.nt-self.ntrans) ))
        ruido = np.zeros((int(self.nt-self.ntrans)))

        #ji = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #jd = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))

        xai = self.xa
        alfai = self.alfa
       
        c1 = self.ctelastica / self.massa
        gm = self.gamma / self.massa

        ie = 0
        it = 0 
        ntc = nt*nc
        
       # for i in range(ns):
        x = 0
        v = 0
        ji1 = 0
        jd1 = 0
        alfai = alfai + self.dalfa
        xai = xai + self.dxa
        xb = - xai
        ua = self.alfa / 2
        ub = self.alfa / 2
        eta = xa

        it =  0
        ig =  0

        while it < ntc :
            if eta == xa:
                teta = - log(1 - np.random.uniform())/ub
                neta = round(teta/dt)
            elif eta == xb:
                teta = - log(1 - np.random.uniform())/ua
                neta = round(teta/dt)
            ie = 0
            while it < neta and it < ntc:
                it = it+1
                ie = ie+1

                u = v
                y = x
                force = -gm * u - c1 * y + eta/massa 
                v = u + force*dt
                x = y + v*dt

                ji1 = ji1 + (v+u) * eta * dt/2
                jd1 = jd1 + gamma*(((v+u)/2)**2 ) * dt 
                ig = ig + 1

                if ig == nc and it > ntrans*nc :
                    xx[int((it/nc)-ntrans)-1] = x
                    vv[int((it/nc)-ntrans)-1] = v
                    ee[int((it/nc)-ntrans)-1] = ji1 - jd1
                    ruido[int((it/nc)-ntrans)-1] = eta
                    ig = 0
                    loading.progress(   ((it/nc) - ntrans)/(nt-ntrans)    )

                if it == ntrans*nc : ig = 0
            if eta == xa:
                eta = xb
            elif eta == xb:
                eta = xa
        self.x = xx
        self.v = vv
        self.e = ee
        self.zeta = ruido
        self.t = np.arange(0, self.nt*self.dt*self.nc, self.dt*self.nc)
        return self

    def draw(self, max_samples,lag):

        self.max_samples = max_samples
        self.x = deque(np.arange(-max_samples-1+lag,lag ), max_samples+ 1)
        self.y = deque(np.zeros(max_samples+1), max_samples+1)

        for i in [0,1]:
            for j in [0,1]:
                self.axs[i,j].set_ylim(0, max_samples)
                self.axs[i,j].set_xlim(lag, max_samples+lag)
                self.axs[i,j].grid()

        self.line00, = self.axs[0, 0].plot(np.array(self.x), np.array(self.y),  'tab:purple')
        self.axs[0, 0].set_title('x(t)')

        self.line01, = self.axs[0, 1].plot(np.array(self.x), np.array(self.y),  'tab:green')
        self.axs[0, 1].set_title('v(t)')

        self.line10, = self.axs[1, 0].plot(np.array(self.x), np.array(self.y),  'tab:blue')
        self.axs[1, 0].set_title('E(t)')

        self.line11, = self.axs[1, 1].plot(np.array(self.x), np.array(self.y),  'tab:red')
        self.axs[1, 1].set_title('$\zeta$(t)')

        self.the_plot.pyplot(plt)

    def drawplotly(self, max_samples,lag):

        self.max_samples = max_samples
        self.x = deque(np.arange(-max_samples-1+lag,lag ), max_samples+ 1)
        self.y = deque(np.zeros(max_samples+1), max_samples+1)

        for i in [0,1]:
            for j in [0,1]:
                self.axs[i,j].set_ylim(0, max_samples)
                self.axs[i,j].set_xlim(lag, max_samples+lag)
                self.axs[i,j].grid()

        self.line00, = self.axs[0, 0].plot(np.array(self.x), np.array(self.y),  'tab:purple')
        self.axs[0, 0].set_title('x(t)')

        self.line01, = self.axs[0, 1].plot(np.array(self.x), np.array(self.y),  'tab:green')
        self.axs[0, 1].set_title('v(t)')

        self.line10, = self.axs[1, 0].plot(np.array(self.x), np.array(self.y),  'tab:blue')
        self.axs[1, 0].set_title('E(t)')

        self.line11, = self.axs[1, 1].plot(np.array(self.x), np.array(self.y),  'tab:red')
        self.axs[1, 1].set_title('$\zeta$(t)')

        self.the_plot.pyplot(plt)
    #plottrajectory(100,0)

    def animate1(self):  # update the y values (every 1000ms)
        
        self.y.append(np.random.randint(self.max_samples)) 
        self.x.append(self.x[-1]+1) 
        
        self.line00.set_ydata(np.array(self.y))
        self.line00.set_xdata(np.array(self.x)) 
    # the_plot.pyplot(plt)

        self.line01.set_ydata(np.array(self.y))
        self.line01.set_xdata(np.array(self.x)) 

        self.line10.set_ydata(np.array(self.y))
        self.line10.set_xdata(np.array(self.x)) 

        self.line11.set_ydata(np.array(self.y))
        self.line11.set_xdata(np.array(self.x)) 

        self.the_plot.pyplot(plt)

    def animate2(self):  # update the y values (every 1000ms)
        self.line.set_ydata(np.array(y))
        self.line.set_xdata(np.array(x))
        self.the_plot.pyplot(plt)
        self.y.append(np.random.randint(max_x)) #append y with a random integer between 0 to 100
        self.x.append(x[-1]+1) #append y with a random integer between 0 to 100

    def clean(self):
        self.the_plot.pyplot(plt.close())
        #self.fig.close()
        #self.fig.clf()

#nt = 1000
#max_samples = 10
#uou = plot_trajectory()
#uou.openframe()
#uou.draw(max_samples, lag = 0)

#for i in range(nt):
#    if not i == 1 and i%max_samples == 1:
#        uou.clean()
#        uou.openframe()
#        uou.draw(max_samples,i)
#    uou.animate1()
    #time.sleep(0.01)

loading = st.progress(0)

##for percent_complete in range(100):
#    time.sleep(0.1)
#    loading.progress(percent_complete + 1)

nt = st.sidebar.number_input('Enter Total running Time', value = 20)
ntrans = st.sidebar.number_input('Enter Transient Timesteps (will be left out of plot)', value = 0)
nc = st.sidebar.number_input('Enter Coarse graining Scale', value = 100)
dt = st.sidebar.number_input('Enter Timestep size', value = 0.001, min_value=0.0001, step=0.001)
massa = st.sidebar.number_input('Enter Mass', value = 1)
gamma = st.sidebar.number_input('Enter dissipation *gamma*', value = 1)
ctelastica = st.sidebar.number_input('Enter harmonic constant *k*', value = 1)
alfa = st.sidebar.number_input('Enter reservoir inverse decay time *alfa*', value = 10)
xa = st.sidebar.number_input('Enter reservoir amplitude *a*', value = 1)

uou = plot_trajectory(nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, 0, xa, 0)

uou = uou.simulate()

uou.xxplotly()


uou.vvplotly()



uou.zetaplotly()



uou.eeplotly()
