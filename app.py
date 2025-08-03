import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.title("TV Delivery Optimizer")

st.markdown("""
**This example of prescriptive analytics solves a linear programming problem with python code, creating a data model as a basis for automation.**  
""")

# Labels and Data
depot_labels = ["D1", "D2", "D3"]
store_labels = ["Store 1", "Store 2", "Store 3"]
store_caps = [2000, 3000, 2000]

distances = np.array([
    [22, 33, 40],
    [27, 30, 22],
    [36, 20, 25],
])
cost_per_mile = 5

# --- Input + Matrix in Columns ---
st.markdown("### 📦 TVs Available at Each Depot")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("input_form"):
        d1_supply = st.number_input("TVs available at Depot D1:", min_value=0, value=2500)
        d2_supply = st.number_input("TVs available at Depot D2:", min_value=0, value=3100)
        d3_supply = st.number_input("TVs available at Depot D3:", min_value=0, value=1250)
        submit = st.form_submit_button("Calculate")

with col2:
    st.markdown("#### 🗺️ Distance Matrix (miles)")
    distance_df = pd.DataFrame(distances, index=depot_labels, columns=store_labels)
    st.dataframe(distance_df.style.format(precision=0), use_container_width=True)

# --- Calculation Section ---
if submit:
    depot_supply = [d1_supply, d2_supply, d3_supply]
    total_supply = sum(depot_supply)
    total_demand = sum(store_caps)

    # --- Chart ---
    st.markdown("### 📊 Supply vs Demand Overview")

    fig, ax = plt.subplots(figsize=(9, 2.5))  # Reduced height

    bar_height = 0.15
    supply_bar_y = 0
    demand_bar_y = 1

    d1, d2, d3 = d1_supply, d2_supply, d3_supply

    ax.barh(supply_bar_y, d1, height=bar_height, color='skyblue', label='D1 Supply')
    ax.barh(supply_bar_y, d2, height=bar_height, color='limegreen', left=d1, label='D2 Supply')
    ax.barh(supply_bar_y, d3, height=bar_height, color='orange', left=d1 + d2, label='D3 Supply')

    ax.barh(demand_bar_y, total_demand, height=bar_height, color='green', label='Total Store Demand')

    ax.set_xticks(np.arange(0, max(total_supply, total_demand) + 1000, 1000))
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    ax.set_yticks([supply_bar_y, demand_bar_y])
    ax.set_yticklabels(['Total Depot Supply', 'Total Store Demand'])
    ax.set_xlabel("Number of TVs")
    ax.set_title("Depot Supply vs Store Demand")
    ax.set_xlim(0, max(total_supply, total_demand) * 1.15)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax.text(total_supply + 50, supply_bar_y, str(total_supply), va='center')
    ax.text(total_demand + 50, demand_bar_y, str(total_demand), va='center')

    st.pyplot(fig)

    # --- Validation ---
    if total_supply > total_demand:
        st.error(f"""
        ❌ **Total depot supply ({total_supply}) exceeds store demand ({total_demand}).**  
        Please adjust depot inputs accordingly to avoid over-delivery.
        """)
        st.stop()

    # --- Display Depot Table ---
    depot_df = pd.DataFrame({"Depot": depot_labels, "TVs Available": depot_supply})
    st.markdown(depot_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
        .hide(axis="index")
        .to_html(), unsafe_allow_html=True)

    # --- Store Capacity Table ---
    store_df = pd.DataFrame({"Store": store_labels, "Capacity": store_caps})
    st.markdown("### 🏬 Store Capacity Constraints")
    st.markdown(store_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
        .hide(axis="index")
        .to_html(), unsafe_allow_html=True)

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

