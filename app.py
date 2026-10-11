import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="SmartStock",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# Small styling improvements
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        .small-note {
            color: #666666;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    decisions = pd.read_csv(
        "smartstock_decision_support.csv"
    )

    forecasts = pd.read_csv(
        "smartstock_forecasts.csv"
    )

    shap_importance = pd.read_csv(
        "smartstock_shap_importance.csv"
    )

    decisions["date"] = pd.to_datetime(
        decisions["date"]
    )

    forecasts["date"] = pd.to_datetime(
        forecasts["date"]
    )

    return decisions, forecasts, shap_importance


decisions, forecasts, shap_importance = load_data()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📦 SmartStock")

st.sidebar.caption(
    "Retail demand forecasting & inventory decision support"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Forecast Explorer",
        "Inventory Decisions",
        "Explain the Model",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Forecast period: Aug–Dec 2024"
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("📦 SmartStock")

    st.subheader(
        "AI-powered retail demand forecasting and inventory decision support"
    )

    st.write(
        """
        SmartStock helps inventory managers understand expected weekly demand
        and identify where stock may need to be increased, maintained or reduced.
        """
    )

    st.info(
        "SmartStock supports inventory decisions rather than automatically placing stock orders."
    )


    # ------------------------------
    # Key metrics
    # ------------------------------

    total_forecasts = len(decisions)

    increase_count = (
        decisions["inventory_action"] == "Increase stock"
    ).sum()

    maintain_count = (
        decisions["inventory_action"] == "Maintain stock"
    ).sum()

    reduce_count = (
        decisions["inventory_action"] == "Reduce stock"
    ).sum()

    clearance_count = (
        decisions["clearance_review"] == "Yes"
    ).sum()


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Forecasts",
        f"{total_forecasts:,}"
    )

    col2.metric(
        "Increase Stock",
        f"{increase_count:,}"
    )

    col3.metric(
        "Reduce Stock",
        f"{reduce_count:,}"
    )

    col4.metric(
        "Clearance Reviews",
        f"{clearance_count:,}"
    )


    # ------------------------------
    # Recommendation summary
    # ------------------------------

    st.subheader("Inventory recommendation summary")

    action_counts = (
        decisions["inventory_action"]
        .value_counts()
        .reindex(
            [
                "Increase stock",
                "Maintain stock",
                "Reduce stock"
            ]
        )
        .reset_index()
    )

    action_counts.columns = [
        "Inventory Action",
        "Number of Forecasts"
    ]


    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        action_counts["Inventory Action"],
        action_counts["Number of Forecasts"]
    )

    ax.set_ylabel("Number of forecasts")
    ax.set_xlabel("Inventory recommendation")

    ax.set_title(
        "SmartStock Inventory Recommendations"
    )

    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------
    # System explanation
    # ------------------------------

    st.subheader("How SmartStock works")

    st.markdown(
        """
        **1. Forecast demand**  
        Historical sales, store characteristics and calendar information are
        used to forecast next week's sales.

        **2. Compare with recent demand**  
        The forecast is compared with the recent 8-week sales average.

        **3. Recommend an inventory action**  
        SmartStock recommends whether stock should be increased, maintained
        or reduced.

        **4. Prioritise manager attention**  
        Larger predicted demand changes are marked as higher priority so
        inventory managers know which decisions should be reviewed first.
        """
    )


    with st.expander(
        "How are inventory recommendations calculated?"
    ):

        st.write(
            """
            **Increase stock:** predicted sales are more than 10% above
            the recent 8-week average.

            **Maintain stock:** predicted sales remain within ±10% of
            the recent 8-week average.

            **Reduce stock:** predicted sales are more than 10% below
            the recent 8-week average.

            **Clearance review:** predicted sales are at least 25%
            below the recent 8-week average.
            """
        )

        st.caption(
            """
            These thresholds are prototype business rules and could be
            adjusted in a real deployment based on inventory costs,
            lead times and manager preferences.
            """
        )


# ==================================================
# FORECAST EXPLORER
# ==================================================

elif page == "Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.write(
        """
        Explore actual and predicted weekly sales for an individual
        Walmart store and department.
        """
    )


    # ------------------------------
    # Store selection
    # ------------------------------

    store_list = sorted(
        forecasts["store_id"].unique()
    )

    selected_store = st.selectbox(
        "Select store",
        store_list
    )


    # ------------------------------
    # Department selection
    # ------------------------------

    departments = sorted(
        forecasts.loc[
            forecasts["store_id"] == selected_store,
            "department"
        ].unique()
    )

    selected_department = st.selectbox(
        "Select department",
        departments
    )


    filtered_forecasts = forecasts[
        (forecasts["store_id"] == selected_store)
        &
        (forecasts["department"] == selected_department)
    ].copy()


    filtered_forecasts = (
        filtered_forecasts
        .sort_values("date")
    )


    st.subheader(
        f"Store {selected_store} — Department {selected_department}"
    )


    # ------------------------------
    # Forecast accuracy for selection
    # ------------------------------

    mae = np.mean(
        np.abs(
            filtered_forecasts["weekly_sales"]
            -
            filtered_forecasts["predicted_weekly_sales"]
        )
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Weeks shown",
        len(filtered_forecasts)
    )

    col2.metric(
        "Average actual sales",
        f"${filtered_forecasts['weekly_sales'].mean():,.0f}"
    )

    col3.metric(
        "Average forecast error",
        f"${mae:,.0f}"
    )


    # ------------------------------
    # Clear actual vs predicted chart
    # ------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        filtered_forecasts["date"],
        filtered_forecasts["weekly_sales"],
        marker="o",
        linewidth=2,
        label="Actual Sales"
    )

    ax.plot(
        filtered_forecasts["date"],
        filtered_forecasts["predicted_weekly_sales"],
        marker="o",
        linewidth=2,
        label="Predicted Sales"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly sales ($)")

    ax.set_title(
        f"Actual vs Predicted Sales — Store {selected_store}, Department {selected_department}"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------
    # Forecast table
    # ------------------------------

    st.subheader("Weekly forecast details")


    display_forecasts = filtered_forecasts[
        [
            "date",
            "weekly_sales",
            "predicted_weekly_sales",
            "forecast_error"
        ]
    ].copy()


    display_forecasts["date"] = (
        display_forecasts["date"]
        .dt.strftime("%d %b %Y")
    )


    display_forecasts.columns = [
        "Date",
        "Actual Sales",
        "Predicted Sales",
        "Forecast Error"
    ]


    st.dataframe(
        display_forecasts.style.format(
            {
                "Actual Sales": "${:,.0f}",
                "Predicted Sales": "${:,.0f}",
                "Forecast Error": "${:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# INVENTORY DECISIONS
# ==================================================

elif page == "Inventory Decisions":

    st.title("📋 Inventory Decisions")

    st.write(
        """
        SmartStock compares forecast demand with the recent 8-week
        sales average and translates the forecast into an inventory action.
        """
    )


    # ------------------------------
    # Filters
    # ------------------------------

    col1, col2, col3 = st.columns(3)


    selected_action = col1.selectbox(
        "Inventory action",
        [
            "All",
            "Increase stock",
            "Maintain stock",
            "Reduce stock"
        ]
    )


    selected_priority = col2.selectbox(
        "Priority",
        [
            "All",
            "High",
            "Medium",
            "Low"
        ]
    )


    clearance_only = col3.checkbox(
        "Clearance review only"
    )


    filtered_decisions = decisions.copy()


    if selected_action != "All":

        filtered_decisions = filtered_decisions[
            filtered_decisions[
                "inventory_action"
            ] == selected_action
        ]


    if selected_priority != "All":

        filtered_decisions = filtered_decisions[
            filtered_decisions[
                "priority"
            ] == selected_priority
        ]


    if clearance_only:

        filtered_decisions = filtered_decisions[
            filtered_decisions[
                "clearance_review"
            ] == "Yes"
        ]


    filtered_decisions = (
        filtered_decisions
        .sort_values(
            "absolute_expected_change",
            ascending=False
        )
    )


    # ------------------------------
    # Summary
    # ------------------------------

    st.metric(
        "Matching decisions",
        f"{len(filtered_decisions):,}"
    )


    # ------------------------------
    # Display table
    # ------------------------------

    decision_table = filtered_decisions[
        [
            "date",
            "store_id",
            "department",
            "predicted_weekly_sales",
            "rolling_mean_8",
            "expected_change_pct",
            "inventory_action",
            "priority",
            "clearance_review"
        ]
    ].copy()


    decision_table["date"] = (
        decision_table["date"]
        .dt.strftime("%d %b %Y")
    )


    decision_table.columns = [
        "Date",
        "Store",
        "Department",
        "Predicted Sales",
        "8-Week Average",
        "Expected Change %",
        "Action",
        "Priority",
        "Clearance Review"
    ]


    st.dataframe(
        decision_table.style.format(
            {
                "Predicted Sales": "${:,.0f}",
                "8-Week Average": "${:,.0f}",
                "Expected Change %": "{:,.1f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------
    # Download filtered decisions
    # ------------------------------

    download_data = decision_table.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download filtered recommendations",
        data=download_data,
        file_name="smartstock_recommendations.csv",
        mime="text/csv"
    )


    with st.expander(
        "How should these recommendations be used?"
    ):

        st.write(
            """
            Higher-priority recommendations indicate larger expected
            changes in demand and should receive more manager attention.

            The system does not automatically determine exact order
            quantities because the available dataset does not contain
            current physical stock levels or individual product units.

            SmartStock therefore provides an evidence-based direction
            and priority for the inventory decision.
            """
        )


# ==================================================
# EXPLAIN MODEL
# ==================================================

elif page == "Explain the Model":

    st.title("🔎 Explain the Model")

    st.write(
        """
        XGBoost SHAP contribution values help explain which features
        have the greatest overall influence on weekly sales forecasts.
        """
    )


    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=20,
        value=10
    )


    top_features = (
        shap_importance
        .head(top_n)
        .sort_values(
            "Mean absolute SHAP"
        )
    )


    # ------------------------------
    # Horizontal SHAP plot
    # ------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        top_features["Feature"],
        top_features["Mean absolute SHAP"]
    )

    ax.set_xlabel(
        "Mean absolute SHAP contribution"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Most Influential Forecasting Features"
    )

    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------
    # Feature table
    # ------------------------------

    st.subheader(
        "Feature importance"
    )


    st.dataframe(
        shap_importance
        .head(top_n)
        .style.format(
            {
                "Mean absolute SHAP": "{:,.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------
    # Explanation
    # ------------------------------

    st.subheader(
        "What does this mean?"
    )


    st.write(
        """
        The **8-week rolling sales average** had the largest overall
        influence on the model, showing that recent demand is especially
        important when forecasting the following week.

        **Store size** was also highly influential because stores with
        different capacities naturally operate at different sales levels.

        **Holiday and calendar information** also affected the forecasts,
        allowing the model to adjust for demand changes around important
        retail periods such as Black Friday.
        """
    )


    st.info(
        """
        A larger mean absolute SHAP value means that the feature
        generally has a stronger influence on the model's predictions.
        """
    )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        """
        The final XGBoost model was selected using a chronological
        validation period before being evaluated on the untouched
        final test period.
        """
    )


    # ------------------------------
    # Main test metrics
    # ------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "MAE",
        "$27,472"
    )

    col2.metric(
        "RMSE",
        "$38,007"
    )

    col3.metric(
        "WAPE",
        "43.46%"
    )

    col4.metric(
        "R²",
        "0.51"
    )


    st.caption(
        "Final test period: 3 August 2024 – 21 December 2024"
    )


    # ------------------------------
    # Baseline improvement
    # ------------------------------

    st.subheader(
        "Improvement over simple forecasting"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "RMSE improvement vs previous week",
        "31.31%"
    )


    col2.metric(
        "RMSE improvement vs 8-week average",
        "12.54%"
    )


    # ------------------------------
    # Why it matters
    # ------------------------------

    st.subheader(
        "Why this matters"
    )


    st.write(
        """
        SmartStock does more than produce a weekly sales prediction.

        The tuned XGBoost model improved on both the simple previous-week
        forecast and the stronger 8-week rolling-average baseline.

        The predictions are then translated into inventory recommendations,
        allowing inventory managers to identify where stock levels may need
        attention.
        """
    )


    st.warning(
        """
        High-priority recommendations should still be reviewed carefully.
        During the historical test period, larger expected demand changes
        were also associated with larger forecasting errors.
        """
    )


    st.subheader(
        "Decision-support principle"
    )

    st.write(
        """
        SmartStock is designed to support human decision-making rather than
        replace the inventory manager. Forecasts, recommendation priority and
        model explanations can be considered together before a final stock
        decision is made.
        """
    )