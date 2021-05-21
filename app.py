import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import log


class SimulacaoBimodal:
    def __init__(self, ns, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa):
        self.ns = ns
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

    def run(self):

        tempo = np.arange(self.dt * self.nc, self.nt * self.dt * self.nc, self.dt * self.nc)
        xx = np.zeros((self.nt-self.ntrans,self.ns ))
        vv = np.zeros((self.nt-self.ntrans,self.ns ))
        ee = np.zeros((self.nt-self.ntrans,self.ns ))
        ji = np.zeros((self.nt-self.ntrans,self.ns ))
        jd = np.zeros((self.nt-self.ntrans,self.ns ))
        ruido = np.zeros((self.nt-self.ntrans,self.ns ))

        xai = self.xa
        alfai = self.alfa
       
        c1 = self.ctelastica / self.massa
        gm = self.gamma / self.massa

        ie = 0
        it = 0 
        
        for i in range(ns):
            x = 0
            v = 0
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

                    u = v
                    y = x
                    force = -gm * u - c1 * y + eta/massa 
                    v = u + force*dt
                    x = y + v*dt

                    ji1 = ji1 + (v+u) * eta * dt/2
                    jd1 = jd1 + gamma*(((v+u)/2)**2 ) * dt 
                    ig = ig + 1

                    if ig == nc and it > ntrans*nc :
                        xx[(it/nc)-ntrans] = x
                        vv[(it/nc)-ntrans] = v
                        ee[(it/nc)-ntrans] = ji1 - jd1
                        ji[(it/nc)-ntrans] = ji1
                        jd[(it/nc)-ntrans] = jd1
                        ruido[(it/nc)-ntrans] = eta
                        ig = 0
                    if it == ntrans*nc : ig =0
                if eta == xa:
                    eta = xb
                elif eta == xb:
                    eta = xa
        return xx, vv, ee, ji, jd, ruido

    def draw(self):

        tempo = np.arange(self.dt * self.nc, self.nt * self.dt * self.nc, self.dt * self.nc)
        #fig, ax = plt.subplots()
        #ax.set_ylim(0, 10)
        #line, = ax.plot(tempo, np.zeros(len(tempo)))
        #the_plot = st.pyplot(plt)
        #def init():  # give a clean slate to start
        #line.set_ydata([np.nan] * self.nt)
        x0 = np.zeros((1,1))
        ruido0 = np.array([[self.xa]])
        e0 = np.zeros((1,1))

        st.text("")
        st.latex("x(t)")
        chartX = st.line_chart(x0)
        st.text("")
        st.latex("\eta(t)")
        chartEta = st.line_chart(ruido0 )

        st.text("")
        st.latex("E(t)")
        chartE = st.line_chart(e0)

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
        
        for i in range(ns):
            x = 0
            v = 0
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

                    u = v
                    y = x
                    force = -gm * u - c1 * y + eta/massa 
                    v = u + force*dt
                    x = y + v*dt

                    ji1 = ji1 + (v+u) * eta * dt/2
                    jd1 = jd1 + gamma*(((v+u)/2)**2 ) * dt 
                    ig = ig + 1

                    if ig == nc and it > ntrans*nc :
                        #xx[int((it/nc)-ntrans)] = x
                        #vv[int((it/nc)-ntrans)] = v
                        #ee[int((it/nc)-ntrans)] = ji1 - jd1
                        #ji[int((it/nc)-ntrans)] = ji1
                        #jd[int((it/nc)-ntrans)] = jd1
                        #ruido[int((it/nc)-ntrans)] = eta
                        #progress_bar.progress(i)
                        chartX.add_rows(np.array([[x]]))
                        chartEta.add_rows(np.array([[eta]]))
                        chartE.add_rows(np.array([[ji1-jd1]]))
                        #st.line_chart(xx)
                        #st.line_chart(ruido)
                        #line.set_xdata(np.arange(1,int((it/nc)-ntrans)+1,1))
                        #line.set_ydata(xx[:int((it/nc)-ntrans)])
                        #the_plot.pyplot(plt)
                        ig = 0
                    if it == ntrans*nc : ig =0
                if eta == xa:
                    eta = xb
                elif eta == xb:
                    eta = xa
        return xx, vv, ee, ji, jd, ruido

#__init__(self, ns, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa):

st.title('Unidimensional particle under dichotomous noise')
st.header('by João Ribeiro')
st.text("")
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

ns = st.sidebar.number_input('Enter Sample Number', value = 100) # max e min
nt = st.sidebar.number_input('Enter Timesteps Number', value = 300)
ntrans = st.sidebar.number_input('Enter Transient Timesteps (will be left out of plot)', value = 0)
nc = st.sidebar.number_input('Enter Coarse graining Scale', value = 100)
dt = st.sidebar.number_input('Enter Timestep size', value = 0.001)
massa = st.sidebar.number_input('Enter Mass', value = 1)
gamma = st.sidebar.number_input('Enter dissipation *gamma*', value = 1)
ctelastica = st.sidebar.number_input('Enter harmonic constant *k*', value = 1)
alfa = st.sidebar.number_input('Enter reservoir inverse decay time *alfa*', value = 2)
xa = st.sidebar.number_input('Enter reservoir amplitude *a*', value = 1)

dalfa = 0
dxa = 0

sim = SimulacaoBimodal(ns, nt, ntrans, nc, dt, massa, gamma, ctelastica, alfa, dalfa, xa, dxa)

sim.draw()
#hash = st.text_input('Scrape Twitter for your target Hashtag! ;)')

st.button("Re-run")


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

#progress_bar.empty()

    

