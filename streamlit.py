import streamlit as st
import pandas as pd
from app import SmartBusinessIntelligence

st.set_page_config(page_title="Advisor System BI", layout="wide")

st.title("Smart Business Intelligence Advisor System")

# Sidebar: Data Upload
st.sidebar.header("Upload Data")
merchant_file = st.sidebar.file_uploader("Upload Merchant Excel (.xlsx)", type=["xlsx"])
competitor_file = st.sidebar.file_uploader("Upload Competitor CSV", type=["csv"])

if merchant_file and competitor_file:
    # Save uploaded files to disk (required for pandas read_excel/read_csv)
    with open("uploaded_merchant.xlsx", "wb") as f:
        f.write(merchant_file.getbuffer())
    with open("uploaded_competitor.csv", "wb") as f:
        f.write(competitor_file.getbuffer())

    # Initialize system
    sbi = SmartBusinessIntelligence()
    sbi.load_data("uploaded_merchant.xlsx")
    sbi.load_competitor_data("uploaded_competitor.csv")

    st.success("Data loaded successfully!")

    # Tabs for each analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Business Summary", "Product Insights", "Customer Insights", "Competitor Analysis", "Market Trends", "Geographic Analysis"
    ])

    with tab1:
        st.header("Business Summary")
        summary = sbi._generate_business_summary()
        st.json(summary)

    with tab2:
        st.header("Product Insights")
        prod = sbi._generate_product_insights()
        st.json(prod)
        if prod["price_distribution"]:
            st.bar_chart(pd.DataFrame.from_dict(prod["price_distribution"], orient='index'))

    with tab3:
        st.header("Customer Insights")
        cust = sbi._generate_customer_insights()
        st.json(cust)

    with tab4:
        st.header("Competitor Analysis")
        comp = sbi.competitor_intelligence()
        if "error" in comp:
            st.error(comp["error"])
        else:
            st.json(comp["competitor_summary"])
            st.write("Top Competitors:")
            st.table(pd.DataFrame(comp["rating_analysis"]["top_competitors"]))

    with tab5:
        st.header("Market Trends")
        trends = sbi.market_trend_observatory()
        st.write("Food Trends:")
        st.table(pd.DataFrame(trends["food_trends"]))
        st.write("Consumer Trends:")
        st.table(pd.DataFrame(trends["consumer_trends"]))
        st.write("Recommendations:")
        st.table(pd.DataFrame(trends["recommendations"]))

    with tab6:
        st.header("Geographic Analysis")
        geo = sbi._generate_geografis_analysis()
        st.json(geo)
        if geo.get("detail"):
            st.write("Detail Penilaian Lokasi:")
            st.table(pd.DataFrame(geo["detail"]))

    # Downloadable report
    if st.button("Generate Full Report"):
        report = sbi.generate_report()
        st.subheader("Executive Summary")
        st.write(report["executive_summary"])
        st.json(report)
else:
    st.info("Please upload both Merchant Excel and Competitor CSV files to start analysis.")