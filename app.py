import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.lp_solver import lp_models_latex
from modules.scenarios import SCENARIOS, get_scenario, scenario_insights
from modules.solve_2x2 import graphical_data
from modules.solver import solve_game
from utils.data_validator import validate_matrix

# ---------------------------------------------------------
# 1. Page configuration & styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; font-style: italic; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Applications of Matrices in Two-Player Zero-Sum Game Theory • Linear Algebra Project</div>',
    unsafe_allow_html=True,
)

REFERENCES = [
    "Osborne, M. J. (2004). *An Introduction to Game Theory*. Oxford University Press.",
    "Taha, H. A. (2017). *Operations Research: An Introduction* (10th ed.). Pearson.",
    "von Neumann, J. (1928). Zur Theorie der Gesellschaftsspiele. *Mathematische Annalen, 100*, 295-320.",
    "Dantzig, G. B. (1951). A proof of the equivalence of the programming problem and the game problem. "
    "In T. C. Koopmans (Ed.), *Activity Analysis of Production and Allocation* (pp. 330-335). Wiley.",
    "Ferguson, T. S. (2014). *Game Theory* (2nd ed.). Lecture notes, Department of Mathematics, UCLA.",
    "Owen, G. (2013). *Game Theory* (4th ed.). Emerald Group Publishing.",
    "Luenberger, D. G., & Ye, Y. (2016). *Linear and Nonlinear Programming* (4th ed.). Springer.",
    "Gordon, L. A., & Loeb, M. P. (2002). The economics of information security investment. "
    "*ACM Transactions on Information and System Security, 5*(4), 438-457.",
    "Tambe, M. (2011). *Security and Game Theory: Algorithms, Deployed Systems, Lessons Learned*. Cambridge University Press.",
]


def short(names):
    return [n.split(":")[0] for n in names]


# ---------------------------------------------------------
# 2. Sidebar: choose the game
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.radio("Input mode", ["Real-world scenario library", "Custom m × n matrix"])

    scenario = None
    if mode.startswith("Real-world"):
        titles = [s.title for s in SCENARIOS]
        chosen = st.selectbox("Choose scenario:", titles)
        scenario = SCENARIOS[titles.index(chosen)]
        A0 = scenario.A.copy()
        row_names, col_names = list(scenario.row_names), list(scenario.col_names)
    else:
        m_in = st.number_input("Rows m (Player 1 strategies):", min_value=2, max_value=8, value=3, step=1)
        n_in = st.number_input("Columns n (Player 2 strategies):", min_value=2, max_value=8, value=3, step=1)
        m_in, n_in = int(m_in), int(n_in)
        A0 = np.zeros((m_in, n_in), dtype=float)
        row_names = [f"X{i + 1}" for i in range(m_in)]
        col_names = [f"Y{j + 1}" for j in range(n_in)]
        st.caption("Player 1 (rows) maximises the payoff a_ij, Player 2 (columns) minimises it.")

m_rows, n_cols = A0.shape

# ---------------------------------------------------------
# 3. Tabs
# ---------------------------------------------------------
tab_solver, tab_theory, tab_lp, tab_apps, tab_refs = st.tabs(
    ["🎯 Solver", "📐 Theory & Algorithms", "🧮 Linear Programming", "🌍 Applications", "📚 References"]
)

# ---------------------------------------------------------
# Helper plots
# ---------------------------------------------------------
def plot_strategy_distribution(probs, player_name, color_scale, labels):
    probs = np.asarray(probs, dtype=float)
    labels = list(labels)
    if len(probs) != len(labels):
        raise ValueError(f"{player_name}: probability count mismatch.")

    df = pd.DataFrame({
        "Strategy": labels,
        "Probability": probs,
        "Label": [f"{value:.4f}" for value in probs],
    })
    fig = px.bar(
        df, x="Strategy", y="Probability", text="Label",
        title=f"{player_name} Optimal Strategy Distribution",
        color="Probability", color_continuous_scale=color_scale,
        category_orders={"Strategy": labels},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        font_family="Times New Roman", title_font_size=16, showlegend=False,
        yaxis=dict(range=[0, 1.25], title="Probability", tickformat=".2f"),
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels),
        coloraxis_showscale=False, margin=dict(t=65, b=50),
    )
    return fig


def plot_envelope(g, player, labels_other):
    """Graphical method: lines for each opponent pure strategy and the lower / upper envelope."""
    fig = go.Figure()
    for k, line in enumerate(g["lines"]):
        fig.add_trace(go.Scatter(x=g["t"], y=line, mode="lines", name=labels_other[k], line=dict(width=1.5)))
    env_name = "Lower envelope (Player 1 maximises)" if player == "row" else "Upper envelope (Player 2 minimises)"
    fig.add_trace(go.Scatter(x=g["t"], y=g["envelope"], mode="lines", name=env_name,
                             line=dict(width=4, dash="dash", color="crimson")))
    fig.add_trace(go.Scatter(x=[g["t_opt"]], y=[g["v_opt"]], mode="markers+text",
                             text=[f"t*={g['t_opt']:.3f}, v={g['v_opt']:.3f}"], textposition="top center",
                             marker=dict(size=12, color="crimson"), name="Optimum"))
    xlabel = "Probability of Player 1's first strategy" if player == "row" else "Probability of Player 2's first strategy"
    fig.update_layout(font_family="Times New Roman", xaxis_title=xlabel, yaxis_title="Expected payoff",
                      title="Graphical method (envelope of expected payoffs)", margin=dict(t=60))
    return fig


# ---------------------------------------------------------
# 4. SOLVER TAB
# ---------------------------------------------------------
with tab_solver:
    st.subheader("📝 Payoff Matrix A = (aᵢⱼ)")
    if scenario is not None:
        st.caption(f"Payoff unit: {scenario.unit}. Rows = {scenario.row_player}, columns = {scenario.col_player}. "
                   "You may edit the numbers; the written analysis in the Applications tab refers to the preset values.")

    default_df = pd.DataFrame(A0, columns=col_names, index=row_names)
    edited_df = st.data_editor(
        default_df,
        key=f"editor_{mode}_{scenario.key if scenario else 'custom'}_{m_rows}x{n_cols}",
        num_rows="fixed",
    )
    solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary")

    ok, msg, A = validate_matrix(edited_df.to_numpy())
    if not ok:
        st.error(msg)
    else:
        signature = (A.shape, tuple(A.flatten().tolist()))
        if solve_btn:
            try:
                st.session_state["report"] = (signature, solve_game(A))
                st.balloons()
            except Exception as exc:  # show the real reason instead of failing silently
                st.session_state.pop("report", None)
                st.error(f"The solver failed: {exc}")

        stored = st.session_state.get("report")
        if stored is None or stored[0] != signature:
            st.info("Press **Calculate Equilibrium Solution** to solve the current matrix.")
        else:
            rep = stored[1]
            sp = rep.saddle
            st.markdown("---")
            st.header("🎯 Equilibrium Results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Value of the game v", f"{rep.value:.4f}")
            c2.metric("Maximin (lower value)", f"{sp['maximin']:.4f}")
            c3.metric("Minimax (upper value)", f"{sp['minimax']:.4f}")
            c4.metric("x* ᵀ A y*", f"{rep.verification['xAy']:.4f}")
            st.caption(f"Method used: **{rep.method}**")

            # --- Step 1: maximin / minimax
            st.markdown("#### Step 1 - Maximin and Minimax principles")
            table = pd.DataFrame(A, index=row_names, columns=col_names)
            table["Row min"] = A.min(axis=1)
            table.loc["Column max"] = list(A.max(axis=0)) + [np.nan]
            st.dataframe(table)
            if sp["has_saddle_point"]:
                pts = ", ".join(f"({row_names[i].split(':')[0]}, {col_names[j].split(':')[0]})" for i, j in sp["saddle_points"])
                st.success(f"✅ Maximin = Minimax = {sp['value']:.4f}. Pure saddle point(s): {pts}.")
            else:
                st.warning(f"⚠️ Maximin ({sp['maximin']:.4f}) < Minimax ({sp['minimax']:.4f}): no pure saddle point, "
                           "so optimal play requires mixed strategies (Minimax Theorem).")

            # --- Step 2: dominance
            with st.expander("Step 2 - Elimination of strictly dominated strategies"):
                for line in rep.dominance["steps"]:
                    st.write("• " + line)
                red = rep.dominance["reduced"]
                st.write(f"Reduced matrix ({red.shape[0]} × {red.shape[1]}):")
                st.dataframe(pd.DataFrame(
                    red,
                    index=[row_names[i] for i in rep.dominance["row_idx"]],
                    columns=[col_names[j] for j in rep.dominance["col_idx"]],
                ))

            # --- Step 3: solution
            st.markdown("#### Step 3 - Optimal strategies")
            if rep.two_by_two is not None:
                for line in rep.two_by_two.steps:
                    st.write("• " + line)

            col1, col2 = st.columns(2)
            col1.plotly_chart(plot_strategy_distribution(rep.x, "Player 1", "Blues", short(row_names)))
            col2.plotly_chart(plot_strategy_distribution(rep.y, "Player 2", "Tealgrn", short(col_names)))

            t1, t2 = st.columns(2)
            t1.dataframe(pd.DataFrame({"Strategy": row_names, "x*": rep.x.round(6)}))
            t2.dataframe(pd.DataFrame({"Strategy": col_names, "y*": rep.y.round(6)}))

            # --- Step 4: verification
            st.markdown("#### Step 4 - Verification of optimality")
            ver = rep.verification
            v1, v2 = st.columns(2)
            v1.write("Player 1's expected payoff against each pure column, (xᵀA)ⱼ — must all be ≥ v:")
            v1.dataframe(pd.DataFrame({"Column": col_names, "(xᵀA)ⱼ": ver["col_payoffs"].round(6)}))
            v2.write("Player 1's expected payoff of each pure row against y*, (Ay)ᵢ — must all be ≤ v:")
            v2.dataframe(pd.DataFrame({"Row": row_names, "(Ay)ᵢ": ver["row_payoffs"].round(6)}))
            if ver["is_optimal"]:
                st.success(f"✅ Verified: min (xᵀA)ⱼ = {ver['p1_guarantee']:.4f} ≥ v = {rep.value:.4f} ≥ "
                           f"max (Ay)ᵢ = {ver['p2_guarantee']:.4f}, and xᵀAy = {ver['xAy']:.4f}.")
            else:
                st.error("❌ The verification failed: (x*, y*) is not an optimal pair.")

            # --- Graphical method (2 x n or m x 2)
            gdata = graphical_data(A)
            if gdata:
                with st.expander("Graphical method (2 × n or m × 2 games)"):
                    if "row" in gdata:
                        st.plotly_chart(plot_envelope(gdata["row"], "row", col_names))
                    if "col" in gdata:
                        st.plotly_chart(plot_envelope(gdata["col"], "col", row_names))

            # --- Cross-check
            st.markdown("#### Step 5 - Cross-check with linear programming")
            cross = pd.DataFrame({
                "Method": ["Main solution", "LP (SciPy HiGHS)", "Own simplex implementation"],
                "Value v": [rep.value, rep.lp.value, rep.simplex.value],
                "Optimal pair verified": [ver["is_optimal"], rep.lp_verification["is_optimal"],
                                          rep.simplex_verification["is_optimal"]],
            })
            st.dataframe(cross)
            if rep.values_agree:
                st.success("✅ All methods give the same value of the game (Minimax Theorem).")
            else:
                st.error("❌ The methods disagree on the value; check the input matrix.")

            if scenario is not None:
                st.markdown("#### Interpretation for the current scenario")
                for line in scenario_insights(row_names, col_names, rep, ""):
                    st.markdown("- " + line)

# ---------------------------------------------------------
# 5. THEORY TAB
# ---------------------------------------------------------
with tab_theory:
    st.header("Matrix games in two-player zero-sum theory")
    st.markdown(r"""
**Payoff matrix.** Player 1 chooses a row $i\in\{1,\dots,m\}$, Player 2 a column $j\in\{1,\dots,n\}$,
and Player 1 receives $a_{ij}$ while Player 2 pays it (zero-sum). All information is stored in
$A=(a_{ij})\in\mathbb R^{m\times n}$.

**Pure and mixed strategies.** A pure strategy is a row (or column). A mixed strategy is a probability vector

$$x\in\Delta_m=\{x\ge 0,\ \mathbf 1^{T}x=1\},\qquad y\in\Delta_n=\{y\ge 0,\ \mathbf 1^{T}y=1\}.$$

The expected payoff of Player 1 is the bilinear form

$$E(x,y)=x^{T}Ay=\sum_{i=1}^{m}\sum_{j=1}^{n}x_i\,a_{ij}\,y_j .$$

**Maximin and Minimax.** The security levels of the two players with pure strategies are

$$\underline v=\max_i\min_j a_{ij}\ \ (\text{Maximin}),\qquad \overline v=\min_j\max_i a_{ij}\ \ (\text{Minimax}),\qquad \underline v\le\overline v .$$

**Saddle point.** An entry $a_{i^*j^*}$ is a saddle point if it is the minimum of its row and the maximum of its column;
equivalently $a_{ij^*}\le a_{i^*j^*}\le a_{i^*j}$ for all $i,j$. A pure saddle point exists **iff** $\underline v=\overline v$;
then $v=a_{i^*j^*}$ and pure strategies are optimal.

**Minimax Theorem (von Neumann, 1928).** For every real matrix $A$,

$$\max_{x\in\Delta_m}\ \min_{y\in\Delta_n}\ x^{T}Ay\;=\;\min_{y\in\Delta_n}\ \max_{x\in\Delta_m}\ x^{T}Ay\;=\;v .$$

Hence every matrix game has a value $v$ and optimal mixed strategies $x^*,y^*$ with $x^{*T}A\ge v\mathbf 1^T$ and $Ay^*\le v\mathbf 1$.
""")

    st.subheader("Algorithm for 2 × 2 games")
    st.markdown(r"""
1. Compute row minima, column maxima, Maximin and Minimax. If they are equal, the saddle point gives pure optimal strategies.
2. Otherwise set $D=(a_{11}+a_{22})-(a_{12}+a_{21})$ ($D\neq 0$ when no saddle point exists) and use the **indifference principle**
   (each player mixes so that the opponent is indifferent between his pure strategies):

$$x_1^*=\frac{a_{22}-a_{21}}{D},\quad x_2^*=1-x_1^*,\qquad y_1^*=\frac{a_{22}-a_{12}}{D},\quad y_2^*=1-y_1^*,\qquad v=\frac{a_{11}a_{22}-a_{12}a_{21}}{D}.$$

3. Graphical check: plot $E_j(t)=a_{1j}t+a_{2j}(1-t)$ and take the maximum of the lower envelope $\min_j E_j(t)$.
""")

    st.subheader("Algorithm for the general m × n game")
    st.markdown(r"""
1. **Saddle test** with Maximin / Minimax.
2. **Dominance reduction.** Delete strictly dominated rows and columns repeatedly (the value and optimal strategies are unchanged).
3. **Linear programming.** Shift $B=A+k\mathbf 1\mathbf 1^T>0$ and solve
   $\max\ \mathbf 1^Tq\ \text{ s.t. }Bq\le\mathbf 1,\ q\ge0$ together with its dual
   $\min\ \mathbf 1^Tp\ \text{ s.t. }B^Tp\ge\mathbf 1,\ p\ge0$ with the simplex method (see the *Linear Programming* tab).
4. **Recover** $x^*=p/\sum p_i,\quad y^*=q/\sum q_j,\quad v=1/z^*-k$.
5. **Verify** $x^{*T}A\ge v\mathbf 1^T$, $Ay^*\le v\mathbf 1$ and $x^{*T}Ay^*=v$.

**Why matrix games and linear programming are equivalent.** The problem "Player 1 maximises his guaranteed payoff" is the LP
$\max\{v:\ A^Tx\ge v\mathbf 1,\ \mathbf 1^Tx=1,\ x\ge0\}$ and Player 2's problem is its **dual**
$\min\{w:\ Ay\le w\mathbf 1,\ \mathbf 1^Ty=1,\ y\ge0\}$. Strong duality gives $v=w$, which is exactly the Minimax Theorem
(Dantzig, 1951); conversely every LP can be written as a symmetric matrix game.
""")

# ---------------------------------------------------------
# 6. LINEAR PROGRAMMING TAB
# ---------------------------------------------------------
with tab_lp:
    st.header("Equivalent linear programming model of the current matrix")
    ok_lp, msg_lp, A_lp = validate_matrix(edited_df.to_numpy())
    if not ok_lp:
        st.error(msg_lp)
    else:
        models = lp_models_latex(A_lp)
        st.markdown("**Direct formulation** — Player 1 (maximiser) and its dual, Player 2 (minimiser):")
        cL, cR = st.columns(2)
        cL.latex(models["direct_player1"])
        cR.latex(models["direct_player2"])

        st.markdown(f"**Normalised formulation** with shift $k={models['shift']:g}$ so that $B=A+k>0$:")
        cL2, cR2 = st.columns(2)
        cL2.markdown("Player 1 (minimisation, dual)")
        cL2.latex(models["normalised_player1"])
        cR2.markdown("Player 2 (maximisation, primal)")
        cR2.latex(models["normalised_player2"])
        st.latex(models["recovery"])

        stored_lp = st.session_state.get("report")
        sig_lp = (A_lp.shape, tuple(A_lp.flatten().tolist()))
        if stored_lp is None or stored_lp[0] != sig_lp:
            st.info("Press **Calculate Equilibrium Solution** in the Solver tab to see the LP solution and simplex tableaux.")
        else:
            rep = stored_lp[1]
            st.subheader("Solution of the LP")
            lpc1, lpc2 = st.columns(2)
            lpc1.markdown("**SciPy / HiGHS**")
            lpc1.write(f"v = {rep.lp.value:.6f}")
            lpc1.dataframe(pd.DataFrame({"x*": rep.lp.x.round(6)}, index=row_names))
            lpc1.dataframe(pd.DataFrame({"y*": rep.lp.y.round(6)}, index=col_names))
            lpc2.markdown("**Own simplex implementation (Bland's rule)**")
            lpc2.write(f"v = {rep.simplex.value:.6f} after {len(rep.simplex.iterations) - 1} pivot(s)")
            lpc2.dataframe(pd.DataFrame({"x*": rep.simplex.x.round(6)}, index=row_names))
            lpc2.dataframe(pd.DataFrame({"y*": rep.simplex.y.round(6)}, index=col_names))

            with st.expander("Simplex tableaux, iteration by iteration"):
                for it_no, it in enumerate(rep.simplex.iterations):
                    title = "Initial tableau" if it["pivot"] is None else (
                        f"Iteration {it_no}: pivot on row {it['pivot'][0] + 1}, "
                        f"column {rep.simplex.labels[it['pivot'][1]]}")
                    st.write(f"**{title}**")
                    st.dataframe(pd.DataFrame(it["tableau"], index=it["basis"] + ["z"],
                                              columns=rep.simplex.labels).round(4))

# ---------------------------------------------------------
# 7. APPLICATIONS TAB
# ---------------------------------------------------------
with tab_apps:
    st.header("Real-world application scenarios")
    st.caption("Each scenario states its context and assumptions, the interpretation of the matrix elements, "
               "the meaning of the optimal strategies and an economic / social analysis.")
    titles_all = [s.title for s in SCENARIOS]
    default_idx = titles_all.index(scenario.title) if scenario is not None else 0
    pick = st.selectbox("Scenario to read:", titles_all, index=default_idx, key="apps_pick")
    sc = SCENARIOS[titles_all.index(pick)]

    st.subheader(sc.title)
    st.markdown(f"**Type of solution illustrated:** {sc.kind}")
    st.markdown("### 1. Context")
    st.write(sc.context)
    st.markdown("### 2. Assumptions")
    for a in sc.assumptions:
        st.markdown("- " + a)
    st.markdown("### 3. Payoff matrix and interpretation of its elements")
    st.write(f"Player 1 (rows): **{sc.row_player}** · Player 2 (columns): **{sc.col_player}** · Unit: {sc.unit}")
    st.dataframe(pd.DataFrame(sc.A, index=sc.row_names, columns=sc.col_names))
    st.write(sc.element_meaning)

    sc_report = solve_game(sc.A)
    st.markdown("### 4. Optimal strategies and their meaning")
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Value of the game v", f"{sc_report.value:.4f}")
    cm2.metric("Maximin", f"{sc_report.saddle['maximin']:.4f}")
    cm3.metric("Minimax", f"{sc_report.saddle['minimax']:.4f}")
    st.write(sc.strategy_meaning)
    for line in scenario_insights(sc.row_names, sc.col_names, sc_report, ""):
        st.markdown("- " + line)
    ca, cb = st.columns(2)
    ca.plotly_chart(plot_strategy_distribution(sc_report.x, sc.row_player, "Blues", short(sc.row_names)),
                    key=f"apps_x_{sc.key}")
    cb.plotly_chart(plot_strategy_distribution(sc_report.y, sc.col_player, "Tealgrn", short(sc.col_names)),
                    key=f"apps_y_{sc.key}")

    st.markdown("### 5. Economic and social analysis")
    for para in sc.analysis:
        st.write(para)
    st.markdown("### 6. Limitations of the model")
    st.write(sc.limitations)
    if sc.references:
        st.caption("Related references: " + "; ".join(sc.references) + " (see the References tab).")

# ---------------------------------------------------------
# 8. REFERENCES TAB
# ---------------------------------------------------------
with tab_refs:
    st.header("References")
    for k, ref in enumerate(REFERENCES, start=1):
        st.markdown(f"{k}. {ref}")
