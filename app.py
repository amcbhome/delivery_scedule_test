import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

import streamlit.components.v1 as components


st.title("Delivery Optimisation")

GA_MEASUREMENT_ID = "13361729112"

ga_script = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""

components.html(ga_script, height=0)


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

# --- Input Section ---
st.markdown("Input the quantity to be delivered:")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("input_form"):
        d1_supply = st.number_input("TVs available at Depot D1:", min_value=0, value=2500)
        d2_supply = st.number_input("TVs available at Depot D2:", min_value=0, value=3100)
        d3_supply = st.number_input("TVs available at Depot D3:", min_value=0, value=1250)
        submit = st.form_submit_button("Calculate")

with col2:
    st.markdown("Distance table")
    distance_df = pd.DataFrame(distances, index=depot_labels, columns=store_labels)
    st.dataframe(distance_df.style.format(precision=0), use_container_width=True)

# --- Logic Section ---
if submit:
    depot_supply = [d1_supply, d2_supply, d3_supply]
    total_supply = sum(depot_supply)
    total_demand = sum(store_caps)

    if total_supply > total_demand:
        st.error(f"""
        **Total depot supply ({total_supply}) exceeds store demand ({total_demand}).**  
        Please adjust depot inputs accordingly to avoid over-delivery.
        """)
        st.stop()

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

    st.markdown("## Optimisation Results")

    if res.success:
        x = np.round(res.x).astype(int).reshape(3, 3)
        shipment_df = pd.DataFrame(x, index=depot_labels, columns=store_labels)
        store_delivery = x.sum(axis=0)

        # Constraint validation
        if any(store_delivery > store_caps):
            st.error("Input values are not within scope: at least one store is oversupplied.")
            st.stop()

        # --- Optimised Schedule Section ---
        st.markdown("###Optimised Schedule")
        schedule_df = pd.DataFrame({
            "Store": store_labels,
            "Delivered": store_delivery,
            "Capacity": store_caps
        })
        st.dataframe(schedule_df.style.set_properties(**{
            "text-align": "center"
        }).hide(axis="index"), use_container_width=True)

        # --- Shipment Matrix Section ---
        st.markdown("#### Shipment Breakdown")
        st.markdown(shipment_df.style.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ]).to_html(), unsafe_allow_html=True)

        # --- Cost ---
        total_cost = int(res.fun)
        st.write(f"### Total Delivery Cost: £{total_cost:,}")
    else:
        st.error("Optimization failed: " + res.message)
