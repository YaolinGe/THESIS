import streamlit as st

def render(): 
    st.markdown("""
                # Data Assimilation
                """)
    # st.markdown("""
    #             ### Covariance matrix in GRFs: Defines spatial correlations for accurate simulations.
    #             """)
    # st.latex(r"""
    #         \begin{equation}
    #             \boldsymbol{\Sigma}=\sigma^2 (1+\phi \boldsymbol{h})\exp(-\phi \boldsymbol{h})
    #         \end{equation}
    #         """)
    st.markdown("""
                ### The measurements at each stage $j=1,\ldots,J$ are modeled by a Gaussian likelihood model 
                """)
    st.latex(r"""
            \begin{equation}
                \boldsymbol{y}_j|\boldsymbol{\xi} \sim N(\boldsymbol{F}_j \boldsymbol{\xi}, \boldsymbol{R}_j), \quad \boldsymbol{F}_j = \boldsymbol{0}^{N_j \times n}, \quad \boldsymbol{F}_j[\boldsymbol{D}_j] = \boldsymbol{1}, \quad \boldsymbol{R}_j = \tau^2 \boldsymbol{I}
            \end{equation}
            """)
    st.markdown("""
                ### Via Bayes' rule, data assimilation at stages $j=1,\ldots J$ can be performed by the following recursive equations:
                """)
    st.latex(r"""
        \begin{equation}
            \boldsymbol{G}_j = \boldsymbol{S}_{j-1} \boldsymbol{F}_j^T (\boldsymbol{F}_j \boldsymbol{S}_{j-1} \boldsymbol{F}_j^T + \boldsymbol{R}_j)^{-1} \\
        \end{equation}
        """)
    st.latex(r"""
            \begin{equation}
                \boldsymbol{m}_j = \boldsymbol{m}_{j-1} + \boldsymbol{G}_j(\boldsymbol{y}_j - \boldsymbol{F}_j\boldsymbol{m}_{j-1}) \\
            \end{equation}
            """)
    st.latex(r"""
            \begin{equation}
                \boldsymbol{S}_j = \boldsymbol{S}_{j-1} - \boldsymbol{G}_j\boldsymbol{F}_j \boldsymbol{S}_{j-1}, 
            \end{equation}
            """)
    st.markdown("""
                ### Where the initial conditions are given by:
                """)
    st.latex(r"""
    \boldsymbol{\mu}_0 = \boldsymbol{\mu}, \quad \boldsymbol{S}_0 = \boldsymbol{\Sigma}
    """)
