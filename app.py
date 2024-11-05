import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import log
import plotly.graph_objects as go
from collections import deque
from src.simulation import SimulacaoBimodal

st.title('Unidimensional particle under dichotomous noise')
st.header('an app by João Ribeiro Medeiros')
st.text("")
st.text("")

st.markdown("The simulation herein presented was studied in these two articles by") 
st.markdown("**João Ribeiro Medeiros** and **Sílvio Manuel Duarte Queirós**:") 

st.write("[Thermostatistics of a damped bimodal particle](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.062145)")
st.write("[Effective temperatures for single particle system under dichotomous noise](https://iopscience.iop.org/article/10.1088/1742-5468/ac014e)")

st.text("")
st.text("")

st.markdown("We have considered a non-equilibrium system with mass $m$, and position $x$, ruled by the following model equation:")
st.latex('m \\frac{d^{2}x\left(t \\right)}{dt^{2}}=-\gamma\,\\frac{dx\left(t\\right)}{dt}-k\,x\left(t\\right)+\zeta_{t}.')

st.markdown("Play around with the model!, notice the mechanical parameters in the sidebar")


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


#st.markdown("The parameter $\gamma$ relates to some type  of friction the system is subjected to, and $\zeta$ is the stochastic force describing the interaction between the particle and the dichotomous reservoir for which we use the Stratonovich interpretation. The confinement is established by the harmonic potential, $k\,x^{2}/2$, which can represent the features of the system or else the action of an optical tweezer --- the behaviour of which is known to be very close to harmonicity. --- that is often used so that the particle-system does not diffuse.")
#st.markdown("Regarding $\zeta_t$ it assumes two symmetric values, $\\zeta_t \\in \\left\\{-a,a\\right\\}$ , hopping between them with the same transition rate, $\\mu$, allowing for a bimodal distribution, ")
#st.latex("f\\left(\\zeta\\right)=\\frac{1}{2}\\,\\delta\\left(\\zeta+a\\right)+\\frac{1}{2}\,\delta\left(\zeta-a\\right)")
#st.markdown("and a coloured correlation function with frequency, $\\alpha=2\,\mu$:")
#st.latex("\\left\\langle \\left \\langle \\zeta \\left(t_{1}\\right) \\, \\, \\zeta\\left(t_{2}\\right)\\right\\rangle \\right\\rangle  = a^{2}\,\mathrm{e}^{-\\alpha\,\left\\vert t_{1}-t_{2}\\right\\vert }")
