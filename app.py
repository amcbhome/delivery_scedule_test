import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.title("TV Delivery Optimizer")

st.markdown("""
**A company wishes to minimise its costs of delivering televisions from three depots (D1, D2 and D3) to three stores (Store 1, Store 2, and Store 3). The cost of delivering one TV is £5 per mile. [Source](https://www.accaglobal.com/uk/en/student/exam-support-resources/professional-exams-study-resources/strategic-business-leader/technical-articles/big-data-sbl.html)**
""")

st.markdown("## Problem Setup")

depot_labels = ["D1", "D2", "D3"]
store_labels = ["Store 1", "Store 2", "Store 3"]
store_caps = [2000, 3000, 2000]

distances = np.array([
    [22, 33, 40],
    [27, 30, 22],
    [36, 20, 25],
])
cost_per_mile = 5

st.markdown("### 📦 TVs Available at Each Depot")

with st.form("input_form"):
    d1_supply = st.number_input("TVs available at Depot D1:", min_value=0, value=2500)
    d2_supply = st.number_input("TVs available at Depot D2:", min_value=0, value=3100)
    d3_supply = st.number_input("TVs available at Depot D3:", min_value=0, value=1250)
    submit = st.form_submit_button("Calculate")

if submit:
    depot_supply = [d1_supply, d2_supply, d3_supply]
    total_supply = sum(depot_supply)
    total_demand = sum(store_caps)

    # --- Matplotlib Horizontal Bar Chart with Breakdown ---
    st.markdown("### 📊 Supply vs Demand Overview")

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_height = 0.25
    y_positions = np.arange(3)

    # Individual depot bars
    ax.barh(y_positions[0], d1_supply, height=bar_height, color='skyblue', label='D1 Supply')
    ax.barh(y_positions[1], d2_supply, height=bar_height, color='limegreen', label='D2 Supply')
    ax.barh(y_positions[2], d3_supply, height=bar_height, color='orange', label='D3 Supply')

    # Total supply bar
    total_bar_y = 3.5
    ax.barh(total_bar_y, total_supply, height=bar_height, color='blue', label='Total Depot Supply')

    # Store demand bar
    demand_bar_y = 4.5
    ax.barh(demand_bar_y, total_demand, height=bar_height, color='green', label='Total Store Demand')

    # Red hatched overlay for excess supply
    if total_supply > total_demand:
        ax.barh(
            total_bar_y,
            total_supply - total_demand,
            left=total_demand,
            height=bar_height,
            color='red',
            hatch='///',
            label='Excess Supply'
        )

    # Y-axis labels
    ax.set_yticks([0, 1, 2, total_bar_y, demand_bar_y])
    ax.set_yticklabels(['Depot D1', 'Depot D2', 'Depot D3', 'Total Supply', 'Total Demand'])

    ax.set_xlabel("Number of TVs")
    ax.set_title("Depot Supply vs Store Demand")
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(total_supply, total_demand) * 1.1)

    # Value labels
    for y, val in zip([0, 1, 2], depot_supply):
        ax.text(val + 50, y, str(val), va='center')
    ax.text(total_supply + 50, total_bar_y, str(total_supply), va='center')
    ax.text(total_demand + 50, demand_bar_y, str(total_demand), va='center')

    st.pyplot(fig)

    # --- Error Trap ---
    if total_supply > total_demand:
        st.error(f"""
        ❌ The total number of TVs available from depots (**{total_supply}**) exceeds the total store demand (**{total_demand}**).
        The red hatched section in the chart indicates the excess supply.
        Please reduce depot inputs accordingly.
        """)
        st.stop()

    # --- Depot Table ---
    depot_df = pd.DataFrame({"Depot": depot_labels, "TVs Available": depot_supply})
    st.markdown(depot_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
        .hide(axis="index")
        .to_html(), unsafe_allow_html=True)

    # --- Store Table ---
    store_df = pd.DataFrame({"Store": store_labels, "Capacity": store_caps})
    st.markdown("### 🏬 Store Capacity Constraints")
    st.markdown(store_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
        .hide(axis="index")
        .to_html(), unsafe_allow_html=True)

    st.markdown("### 🗺️ Distance Matrix (miles)")
    distance_df = pd.DataFrame(distances, index=depot_labels, columns=store_labels)
    st.markdown(distance_df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ]).to_html(), unsafe_allow_html=True)

    # --- Optimization ---
    c = (distances * cost_per_mile).flatten()

    A_store = np.zeros((3, 9))
    for j in range(3):
        for i in range(3):
            A_store[j, 3*i + j] = 1
    b_store = store_caps

    A_depot = np.zeros((3, 9))
    for i in range(3):
        A_depot[i, 3*i : 3*i+3] = 1
    b_depot = depot_supply

    bounds = [(0, None) for _ in range(9)]

    res = linprog(
        c=c,
        A_ub=A_store,
        b_ub=b_store,
        A_eq=A_depot,
        b_eq=b_depot,
        bounds=bounds,
        method="highs"
    )

    st.markdown("## Optimization Results")

    if res.success:
        x = np.round(res.x).astype(int).reshape(3, 3)
        shipment_df = pd.DataFrame(x, index=depot_labels, columns=store_labels)

        st.markdown("### ✅ Optimized TV Shipment Plan")
        st.markdown(shipment_df.style.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ]).to_html(), unsafe_allow_html=True)

        total_cost = res.fun
        st.write(f"### 💰 Total Delivery Cost: £{total_cost:,.2f}")
    else:
        st.error("❌ Optimization failed: " + res.message)
