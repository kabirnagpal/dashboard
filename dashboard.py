import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_data():
    st.header("Upload Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        influencers = st.sidebar.file_uploader(
            "Influencers CSV", type=["csv"], key="influencers")
    with col2:
        posts = st.sidebar.file_uploader(
            "Posts CSV", type=["csv"], key="posts")
    with col3:
        tracking = st.sidebar.file_uploader(
            "Tracking Data CSV", type=["csv"], key="tracking")
    with col4:
        payouts = st.sidebar.file_uploader(
            "Payouts CSV", type=["csv"], key="payouts")
    dfs = {}
    if influencers:
        dfs['influencers'] = pd.read_csv(influencers)
    if posts:
        dfs['posts'] = pd.read_csv(posts)
    if tracking:
        dfs['tracking'] = pd.read_csv(tracking)
    if payouts:
        dfs['payouts'] = pd.read_csv(payouts)
    return dfs


def get_name_mapping(dfs):
    return dfs['influencers'].set_index('influencer_id')['name'].to_dict()


def get_date_range_options(dfs, influencer_ids):
    posts = dfs['posts']
    tracking = dfs['tracking']
    posts_dates = pd.to_datetime(
        posts[posts['influencer_id'].isin(influencer_ids)]['date'])
    tracking_dates = pd.to_datetime(
        tracking[tracking['influencer_id'].isin(influencer_ids)]['date'])
    combined_dates = pd.concat([posts_dates, tracking_dates])
    min_date = combined_dates.min()
    max_date = combined_dates.max()
    return (min_date if pd.notnull(min_date) else pd.Timestamp.today()), (max_date if pd.notnull(max_date) else pd.Timestamp.today())


def apply_filters(dfs):
    inf = dfs['influencers']
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        platform = st.multiselect(
            "Platform", ['All'] + sorted(inf['platform'].dropna().unique()), default=['All'])
    with col2:
        category = st.multiselect(
            "Category", ['All'] + sorted(inf['category'].dropna().unique()), default=['All'])
    with col3:
        gender = st.multiselect(
            "Gender", ['All'] + sorted(inf['gender'].dropna().unique()), default=['All'])
    with col4:
        name = st.multiselect("Influencer Name", [
                              'All'] + sorted(inf['name'].dropna().unique()), default=['All'])
    if 'All' not in platform:
        inf = inf[inf['platform'].isin(platform)]
    if 'All' not in category:
        inf = inf[inf['category'].isin(category)]
    if 'All' not in gender:
        inf = inf[inf['gender'].isin(gender)]
    if 'All' not in name:
        inf = inf[inf['name'].isin(name)]
    return inf, col5


def filter_by_date(dfs, influencer_ids, col):
    min_date, max_date = get_date_range_options(dfs, influencer_ids)
    selected_range = col.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(selected_range, (tuple, list)):
        start_date, end_date = pd.to_datetime(
            selected_range[0]), pd.to_datetime(selected_range[-1])
    else:
        start_date = end_date = pd.to_datetime(selected_range)
    return start_date, end_date


def filter_time_range(df, date_col, start_date, end_date):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]


def get_kpis(dfs, filtered_inf, date_start, date_end):
    posts = dfs['posts']
    tracking = dfs['tracking']
    payouts = dfs['payouts']
    influencers_ids = filtered_inf['influencer_id']
    posts_filt = posts[posts['influencer_id'].isin(influencers_ids)]
    tracking_filt = tracking[tracking['influencer_id'].isin(influencers_ids)]
    payouts_filt = payouts[payouts['influencer_id'].isin(influencers_ids)]
    posts_filt = filter_time_range(posts_filt, 'date', date_start, date_end)
    tracking_filt = filter_time_range(
        tracking_filt, 'date', date_start, date_end)
    total_reach = posts_filt['reach'].sum()
    total_orders = tracking_filt['orders'].sum()
    total_revenue = tracking_filt['revenue'].sum()
    total_payout = payouts_filt['total_payout'].sum()
    total_likes = posts_filt['likes'].sum()
    total_comments = posts_filt['comments'].sum()
    avg_engagement = ((total_likes + total_comments) /
                      total_reach * 100) if total_reach > 0 else 0
    roas = (total_revenue / total_payout) if total_payout > 0 else np.nan
    kpis = {
        "Total Reach": total_reach,
        "Total Orders": total_orders,
        "Total Revenue": total_revenue,
        "Total Payout": total_payout,
        "ROAS (Revenue / Spend)": roas,
        "Avg Engagement Rate (%)": avg_engagement,
        "Post Count": posts_filt.shape[0],
        "Unique Campaigns": len(tracking_filt['campaign'].unique())
    }
    return kpis, posts_filt, tracking_filt, payouts_filt


def show_best_worst_kpis(tracking_filt, name_map):
    kpi_cols = st.columns(4)
    product_grp = tracking_filt.groupby('product')['revenue'].sum()
    if not product_grp.empty:
        best_product = product_grp.idxmax()
        best_product_val = product_grp.max()
        worst_product = product_grp.idxmin()
        worst_product_val = product_grp.min()
        kpi_cols[0].metric("Best Product (Revenue)",
                           best_product, f"${best_product_val:,.0f}")
        kpi_cols[1].metric("Worst Product (Revenue)",
                           worst_product, f"${worst_product_val:,.0f}")
    infl_grp = tracking_filt.groupby('influencer_id')['revenue'].sum()
    if not infl_grp.empty:
        best_infl_id = infl_grp.idxmax()
        best_infl_val = infl_grp.max()
        worst_infl_id = infl_grp.idxmin()
        worst_infl_val = infl_grp.min()
        best_infl = name_map.get(best_infl_id, str(best_infl_id))
        worst_infl = name_map.get(worst_infl_id, str(worst_infl_id))
        kpi_cols[2].metric("Best Influencer (Revenue)",
                           best_infl, f"${best_infl_val:,.0f}")
        kpi_cols[3].metric("Worst Influencer (Revenue)",
                           worst_infl, f"${worst_infl_val:,.0f}")


def plot_orders_revenue_forecast(tracking, start_date, end_date):
    tracking = tracking.copy()
    tracking['date'] = pd.to_datetime(tracking['date'])
    tracking = tracking[(tracking['date'] >= pd.to_datetime(start_date)) & (
        tracking['date'] <= pd.to_datetime(end_date))]
    df = tracking.groupby(tracking['date'].dt.date).agg(
        {'orders': 'sum', 'revenue': 'sum'}).reset_index().sort_values('date')
    df['date'] = pd.to_datetime(df['date'])
    N = 90
    forecast_days = 90
    if df.shape[0] < N:
        st.info(f"Not enough data for {N}-day rolling weighted mean forecast.")
        return
    weights = np.arange(1, N+1)
    weights = weights / weights.sum()
    orders_actual = df['orders'].values
    revenue_actual = df['revenue'].values
    orders_roll = [np.nan] * (N-1)
    revenue_roll = [np.nan] * (N-1)
    for i in range(N-1, len(df)):
        orders_roll.append(np.dot(orders_actual[i-N+1:i+1], weights))
        revenue_roll.append(np.dot(revenue_actual[i-N+1:i+1], weights))
    df['orders_weighted_mean'] = orders_roll
    df['revenue_weighted_mean'] = revenue_roll
    last_orders = orders_actual[-N:].tolist()
    last_revenue = revenue_actual[-N:].tolist()
    forecast_orders = []
    forecast_revenue = []
    for i in range(forecast_days):
        next_order = np.dot(last_orders[-N:], weights)
        next_revenue = np.dot(last_revenue[-N:], weights)
        forecast_orders.append(next_order)
        forecast_revenue.append(next_revenue)
        last_orders.append(next_order)
        last_revenue.append(next_revenue)
    last_date = df['date'].max()
    forecast_dates = [last_date +
                      pd.Timedelta(days=i+1) for i in range(forecast_days)]
    actual_df = df.dropna(
        subset=['orders_weighted_mean', 'revenue_weighted_mean'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_df['date'], y=actual_df['orders_weighted_mean'],
        name='Orders Weighted Mean (90d)', mode='lines+markers', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(
        x=actual_df['date'], y=actual_df['revenue_weighted_mean'],
        name='Revenue Weighted Mean (90d)', mode='lines+markers', line=dict(color='orange'), yaxis='y2'))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_orders,
        name='Orders 90-day Forecast', mode='lines', line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_revenue,
        name='Revenue 90-day Forecast', mode='lines', line=dict(dash='dash', color='orange'), yaxis='y2'))
    fig.update_layout(
        title='Orders and Revenue: 90-day Weighted Rolling Mean & 90-Day Recursive Forecast',
        xaxis_title='Date',
        yaxis=dict(title='Orders'),
        yaxis2=dict(title='Revenue', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=12)),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_top_influencers(tracking, name_map, start_date, end_date):
    tracking = filter_time_range(tracking, 'date', start_date, end_date)
    df = tracking.groupby('influencer_id').agg(
        {'revenue': 'sum', 'orders': 'sum'}).reset_index()
    df['Influencer'] = df['influencer_id'].map(name_map)
    top = df.sort_values('revenue', ascending=False).head(10)
    fig = px.bar(top, x='Influencer', y='revenue',
                 color='revenue',
                 labels={'revenue': 'Total Revenue',
                         'Influencer': 'Influencer'},
                 title="Top 10 Influencers by Revenue",
                 hover_data=["orders"],
                 color_continuous_scale='Blues')
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_engagement_time(posts, name_map, start_date, end_date):
    posts = posts.copy()
    posts['date'] = pd.to_datetime(posts['date'])
    posts = posts[(posts['date'] >= pd.to_datetime(start_date))
                  & (posts['date'] <= pd.to_datetime(end_date))]
    posts['engagement_rate'] = (
        (posts['likes'] + posts['comments']) / posts['reach']) * 100
    posts['engagement_rate'] = posts['engagement_rate'].apply(
        lambda x: min(x, 100))
    posts['Month_Year'] = posts['date'].dt.to_period('M').astype(str)
    fig = px.box(
        posts,
        x='Month_Year',
        y='engagement_rate',
        points=False,
        labels={'Month_Year': "Month",
                'engagement_rate': "Engagement Rate (%)"},
        title="Monthly Violin Plot of Engagement Rates per Post"
    )
    fig.update_layout(
        yaxis_title="Engagement Rate (%)",
        xaxis_tickangle=-90,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_payout_efficiency(tracking, payouts, name_map, start_date, end_date):
    tracking = filter_time_range(tracking, 'date', start_date, end_date)
    grouped_revenue = tracking.groupby('influencer_id')[
        'revenue'].sum().reset_index()
    grouped_payout = payouts.groupby('influencer_id')[
        'total_payout'].sum().reset_index()
    merged = pd.merge(grouped_revenue, grouped_payout,
                      on='influencer_id', how='inner')
    merged['ROAS'] = merged['revenue'] / (merged['total_payout'] + 1e-6)
    merged['Influencer'] = merged['influencer_id'].map(name_map)
    fig = px.scatter(merged, x='total_payout', y='revenue',
                     color='ROAS', size='ROAS', hover_name='Influencer',
                     labels={'total_payout': 'Total Payout',
                             'revenue': 'Total Revenue', 'ROAS': 'ROAS'},
                     color_continuous_scale='bluered_r',
                     title="Influencer Revenue vs. Payout (Color & Size = ROAS)")
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_product_revenue_bar(tracking, start_date, end_date):
    tracking = filter_time_range(tracking, 'date', start_date, end_date)
    prod_rev = tracking.groupby('product')['revenue'].sum(
    ).sort_values(ascending=False).reset_index()
    fig = px.bar(prod_rev, x='product', y='revenue',
                 color='revenue', title="Revenue by Product",
                 labels={'revenue': 'Total Revenue', 'product': 'Product'},
                 color_continuous_scale='viridis')
    fig.update_layout(xaxis_tickangle=-90, height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_orders_time_by_product(tracking, start_date, end_date):
    tracking = filter_time_range(tracking, 'date', start_date, end_date)
    tracking['date'] = pd.to_datetime(tracking['date'])
    df = tracking.groupby([tracking['date'].dt.to_period('M').astype(str), 'product'])[
        'orders'].sum().reset_index()
    fig = px.bar(
        df, x='date', y='orders', color='product',
        title="Orders per Month by Product",
        labels={'date': 'Month', 'orders': 'Orders', 'product': 'Product'},
        barmode='stack', height=400)
    fig.update_layout(xaxis_tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(layout="wide")
    st.title("Influencer Campaign and Payout Dashboard")
    dfs = load_data()
    if len(dfs) < 4:
        st.info("Upload all four CSVs to begin.")
        return

    filtered_inf, date_col = apply_filters(dfs)
    if filtered_inf.empty:
        st.warning("No matching influencers after filtering.")
        return

    name_map = get_name_mapping(dfs)
    influencers_ids = filtered_inf['influencer_id']
    date_start, date_end = filter_by_date(dfs, influencers_ids, col=date_col)
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)

    kpis, posts_filt, tracking_filt, payouts_filt = get_kpis(
        dfs, filtered_inf, date_start, date_end)

    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"${kpis['Total Revenue']:,.0f}")
    col2.metric("ROAS", f"{kpis['ROAS (Revenue / Spend)']:.2f}")
    col3.metric("Orders", kpis['Total Orders'])
    col4, col5, col6 = st.columns(3)
    col4.metric("Reach", kpis['Total Reach'])
    col5.metric("Avg Engagement %", f"{kpis['Avg Engagement Rate (%)']:.1f}")
    col6.metric("Total Payout", f"${kpis['Total Payout']:,.0f}")

    show_best_worst_kpis(tracking_filt, name_map)

    st.header("Visualizations")
    with st.expander("Orders & Revenue Over Time with Forecast", expanded=True):
        plot_orders_revenue_forecast(tracking_filt, date_start, date_end)
    with st.expander("Top Influencers (Revenue)", expanded=True):
        plot_top_influencers(tracking_filt, name_map, date_start, date_end)
    with st.expander("Monthly Violin Plot of Engagement Rates per Post", expanded=True):
        plot_engagement_time(posts_filt, name_map, date_start, date_end)
    with st.expander("Payout Efficiency (ROAS)", expanded=True):
        plot_payout_efficiency(tracking_filt, payouts_filt,
                               name_map, date_start, date_end)

    with st.expander("Revenue by Product (Bar Chart)", expanded=False):
        plot_product_revenue_bar(tracking_filt, date_start, date_end)
    with st.expander("Orders per Month by Product (Stacked Bar)", expanded=False):
        plot_orders_time_by_product(tracking_filt, date_start, date_end)

    st.header("Detailed Data")
    with st.expander("Show Filtered Influencers Table"):
        st.dataframe(filtered_inf.drop(
            columns=['influencer_id'], errors='ignore'))
    with st.expander("Show Posts Table"):
        posts_display = posts_filt.copy()
        posts_display['Influencer'] = posts_display['influencer_id'].map(
            name_map)
        st.dataframe(posts_display.drop(
            columns=['influencer_id'], errors='ignore'))
    with st.expander("Show Orders & Revenue Table"):
        tracking_display = tracking_filt.copy()
        tracking_display['Influencer'] = tracking_display['influencer_id'].map(
            name_map)
        st.dataframe(tracking_display.drop(
            columns=['influencer_id'], errors='ignore'))
    with st.expander("Show Payout Info Table"):
        payouts_display = payouts_filt.copy()
        payouts_display['Influencer'] = payouts_display['influencer_id'].map(
            name_map)
        st.dataframe(payouts_display.drop(
            columns=['influencer_id'], errors='ignore'))


if __name__ == "__main__":
    main()
