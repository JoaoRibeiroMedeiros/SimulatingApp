import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import log
import plotly.graph_objects as go
from collections import deque

#import matplotlib 
#matplotlib.use('Qt4Agg')

class SimulacaoBimodal:
    def __init__(self, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa):
        #self.ns = ns
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

    def xxplotly(self):
        fig = go.Figure(
            data=[go.Scatter(x = [0], y = [0],
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
                    line=dict(color="blue", width=2))
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
                    line=dict(color="green", width=2))
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
                    line=dict(color="purple", width=2))
                ]) for k in range(self.nt)]
        )
        #fig.show()
        st.write(fig)


    def openframe(self, lag):
        left  = 0.125  # the left side of the subplots of the figure
        right = 1.0    # the right side of the subplots of the figure
        bottom = 2.0   # the bottom of the subplots of the figure
        top = 2.5      # the top of the subplots of the figure
        wspace = 2   # the amount of width reserved for blank space between subplots
        hspace = 0.5 

        self.fig, self.axs = plt.subplots(4)
        #self.fig.canvas.draw()
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        self.fig.set_size_inches(18.5, 30.5)
        self.the_plot = st.pyplot(plt)

        self.max_samples = self.nt

        #tempo = np.arange(self.dt * self.nc, self.nt * self.dt * self.nc, self.dt * self.nc)
        self.t = deque(np.arange(-self.nt*self.dt*self.nc-1+lag,lag, self.dt*self.nc ), self.nt+ 1)
        self.x = deque(np.zeros(self.nt+1), self.nt+1)
        self.v = deque(np.zeros(self.nt+1), self.nt+1)
        self.zeta = deque(np.zeros(self.nt+1), self.nt+1)
        self.E = deque(np.zeros(self.nt+1), self.nt+1)

        for i in [0,1,2,3]:
            #self.axs[i].set_ylim(0, self.nt)
            self.axs[i].set_xlim(lag, self.nt+lag)
            self.axs[i].grid()

        self.axs[0].set_ylim(- self.xa, self.xa)
        self.axs[1].set_ylim(- self.xa, self.xa)
        self.axs[2].set_ylim(- 1.5*self.xa, 1.5*self.xa)
        self.axs[3].set_ylim(0, self.xa**2)

        self.line00, = self.axs[0].plot(np.array(self.t), np.array(self.x),  'tab:purple')
        self.axs[0].set_title('x(t)', fontsize=20)

        self.line01, = self.axs[1].plot(np.array(self.t), np.array(self.v),  'tab:green')
        self.axs[1].set_title('v(t)', fontsize=20)

        self.line10, = self.axs[2].plot(np.array(self.t), np.array(self.zeta),  'tab:blue')
       # self.axs[2].set_title('$\eta$(t)', fontsize=20)
        self.axs[2].set_title('\zeta(t)', fontsize=20)


        self.line11, = self.axs[3].plot(np.array(self.t), np.array(self.E),  'tab:red')
        self.axs[3].set_title('E(t)', fontsize=20)

        self.the_plot.pyplot(plt)


    def animate(self):

        #tempo = np.arange(self.dt * self.nc, self.nt * self.dt * self.nc, self.dt * self.nc)
        #x0 = np.zeros((1,1))
        #ruido0 = np.array([[self.xa]])
        #e0 = np.zeros((1,1))

        #st.text("")
        #st.latex("x(t)")
        #chartX = st.line_chart(x0)
        #st.text("")
        #st.latex("\eta(t)")
        #chartEta = st.line_chart(ruido0 )

        #st.text("")
        #st.latex("E(t)")
        #chartE = st.line_chart(e0)

        #xx = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #vv = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #ee = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #ji = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #jd = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))
        #ruido = np.zeros((int(self.nt-self.ntrans),int(self.ns) ))

        xai = self.xa
        alfai = self.alfa
       
        c1 = self.ctelastica / self.massa
        gm = self.gamma / self.massa

        ie = 0
        it = 0 
        ntc = nt*nc
        
       # for i in range(ns):
        xx = 0
        vv = 0
        ji1 = 0
        jd1 = 0
        alfai = alfai + dalfa
        xai = xai + dxa
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

                u = vv
                y = xx
                force = -gm * u - c1 * y + eta/massa 
                vv = u + force*dt
                xx = y + vv*dt

                ji1 = ji1 + (vv+u) * eta * dt/2
                jd1 = jd1 + gamma*(((vv+u)/2)**2 ) * dt 
                ig = ig + 1

                if ig == nc and it > ntrans*nc :
                    #xx[int((it/nc)-ntrans)] = x
                    #vv[int((it/nc)-ntrans)] = v
                    #ee[int((it/nc)-ntrans)] = ji1 - jd1
                    #ji[int((it/nc)-ntrans)] = ji1
                    #jd[int((it/nc)-ntrans)] = jd1
                    #ruido[int((it/nc)-ntrans)] = eta
                    #progress_bar.progress(i)
                    #chartX.add_rows(np.array([[x]]))
                    #chartEta.add_rows(np.array([[eta]]))
                    #chartE.add_rows(np.array([[ji1-jd1]]))
                    #st.line_chart(xx)
                    #st.line_chart(ruido)
                    #line.set_xdata(np.arange(1,int((it/nc)-ntrans)+1,1))
                    #line.set_ydata(xx[:int((it/nc)-ntrans)])
                    #the_plot.pyplot(plt)
                    #print(xx)
                    
                    self.x.append(xx) 

                    #print(self.t)
                    #print(self.x)

                    self.v.append(vv) 
                    self.E.append(ji1-jd1) 
                    self.zeta.append(eta) 
                    self.t.append(self.t[-1] + self.dt * self.nc) 
                    
                    self.line00.set_ydata(np.array(self.x))
                    self.line00.set_xdata(np.array(self.t)) 

                    self.line01.set_ydata(np.array(self.v))
                    self.line01.set_xdata(np.array(self.t)) 

                    self.line10.set_ydata(np.array(self.zeta))
                    self.line10.set_xdata(np.array(self.t)) 

                    self.line11.set_ydata(np.array(self.E))
                    self.line11.set_xdata(np.array(self.t)) 

                    #self.axs[0].draw_artist(self.axs[0].patch)
                    #self.axs[0].draw_artist(self.line00)

                    #self.axs[1].draw_artist(self.axs[1].patch)
                    #self.axs[1].draw_artist(self.line01)

                    #self.axs[2].draw_artist(self.axs[2].patch)
                    #self.axs[2].draw_artist(self.line10)

                    #self.axs[3].draw_artist(self.axs[3].patch)
                    #self.axs[3].draw_artist(self.line11)

                    self.the_plot.pyplot(plt)       # -> isso funciona, mas lento

                    #time.sleep(0.01)
                    ig = 0

                if it == ntrans*nc : ig = 0
            if eta == xa:
                eta = xb
            elif eta == xb:
                eta = xa
        #return xx, vv, ji, jd, ruido
        return True

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
#__init__(self, ns, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa):

st.title('Unidimensional particle under dichotomous noise')
st.header('an app by Jo??o Ribeiro Medeiros')
st.text("")
st.text("")

st.markdown("The simulation herein presented was studied in these two articles by") 
st.markdown("**Jo??o Ribeiro Medeiros** and **S??lvio Manuel Duarte Queir??s**:") 

st.write("[Thermostatistics of a damped bimodal particle](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.062145)")
st.write("[Effective temperatures for single particle system under dichotomous noise](https://iopscience.iop.org/article/10.1088/1742-5468/ac014e)")

st.text("")
st.text("")

st.markdown("We have considered a non-equilibrium system with mass $m$, and position $x$, ruled by the following model equation:")
st.latex('m \\frac{d^{2}x\left(t \\right)}{dt^{2}}=-\gamma\,\\frac{dx\left(t\\right)}{dt}-k\,x\left(t\\right)+\zeta_{t}.')
st.markdown("The parameter $\gamma$ relates to some type  of friction the system is subjected to, and $\zeta$ is the stochastic force describing the interaction between the particle and the dichotomous reservoir for which we use the Stratonovich interpretation. The confinement is established by the harmonic potential, $k\,x^{2}/2$, which can represent the features of the system or else the action of an optical tweezer --- the behaviour of which is known to be very close to harmonicity. --- that is often used so that the particle-system does not diffuse.")

st.markdown("Regarding $\zeta_t$ it assumes two symmetric values, $\\zeta_t \\in \\left\\{-a,a\\right\\}$ , hopping between them with the same transition rate, $\\mu$, allowing for a bimodal distribution, ")
st.latex("f\\left(\\zeta\\right)=\\frac{1}{2}\\,\\delta\\left(\\zeta+a\\right)+\\frac{1}{2}\,\delta\left(\zeta-a\\right)")

st.markdown("and a coloured correlation function with frequency, $\\alpha=2\,\mu$:")

st.latex("\\left\\langle \\left \\langle \\zeta \\left(t_{1}\\right) \\, \\, \\zeta\\left(t_{2}\\right)\\right\\rangle \\right\\rangle  = a^{2}\,\mathrm{e}^{-\\alpha\,\left\\vert t_{1}-t_{2}\\right\\vert }")


st.text("")
st.text("")
st.text("")

#ns = st.sidebar.number_input('Enter Sample Number', value = 100) # max e min
nt = st.sidebar.number_input('Enter Total running Time', value = 20, max_value = 30, min_value = 1)
ntrans = st.sidebar.number_input('Enter Transient Timesteps (will be left out of plot)', value = 0, max_value = 10)
nc = st.sidebar.number_input('Enter Coarse graining Scale', value = 10, max_value = 30, min_value = 1)
dt = st.sidebar.number_input('Enter Timestep size', value = 0.01, min_value=0.0001, step=0.001)
massa = st.sidebar.number_input('Enter Mass', value = 1.0,  min_value = 0.1, max_value = 30.0)
gamma = st.sidebar.number_input('Enter dissipation *gamma*', value = 1.0, min_value = 0.1, max_value = 30.0)
ctelastica = st.sidebar.number_input('Enter harmonic constant *k*', value = 1.0, min_value = 0.1, max_value = 30.0)
alfa = st.sidebar.number_input('Enter reservoir inverse decay time *alfa*', value = 10.0, min_value = 0.1, max_value = 30.0)
xa = st.sidebar.number_input('Enter reservoir amplitude *a*', value = 1.0, min_value = 0.1, max_value =20.0)

st.button("Re-run")

loading = st.progress(0)

dalfa = 0
dxa = 0

sim = SimulacaoBimodal( nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa)

sim = sim.simulate()

sim.xxplotly()

sim.vvplotly()

sim.zetaplotly()

sim.eeplotly()


#sim.openframe(0)
#sim.animate()

#hash = st.text_input('Scrape Twitter for your target Hashtag! ;)')

#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

#for i in range(1, 101):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#    status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
#    progress_bar.progress(i)
#    last_rows = new_rows
#    time.sleep(0.05)

# progress_bar.empty()