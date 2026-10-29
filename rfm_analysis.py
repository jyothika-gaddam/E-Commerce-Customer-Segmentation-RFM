#!/usr/bin/env python3
"""Customer segmentation using Recency, Frequency and Monetary (RFM) analysis."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def find_column(df, options, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {options}. Found: {list(df.columns)}")
    return None

def safe_quantile_score(series, ascending=True):
    """Return integer scores 1..5; ties are handled without qcut duplicate-bin errors."""
    ranks = series.rank(method="average", pct=True, ascending=ascending)
    scores = (ranks * 5).apply(lambda x: min(5, max(1, int(__import__('math').ceil(x)))))
    return scores.astype(int)

def build_rfm(input_path):
    df = pd.read_csv(input_path, encoding_errors="replace")
    invoice_col = find_column(df, ["InvoiceNo", "Invoice", "OrderID"])
    quantity_col = find_column(df, ["Quantity", "Qty"])
    date_col = find_column(df, ["InvoiceDate", "Invoice Date", "OrderDate", "Date"])
    price_col = find_column(df, ["UnitPrice", "Price", "Unit Price"])
    customer_col = find_column(df, ["CustomerID", "Customer Id", "Customer ID", "Customer"])
    df = df.rename(columns={invoice_col:"InvoiceNo", quantity_col:"Quantity", date_col:"InvoiceDate",
                            price_col:"UnitPrice", customer_col:"CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["CustomerID","InvoiceNo","Quantity","UnitPrice","InvoiceDate"])
    df = df[(df["CustomerID"].str.lower() != "nan") & (df["CustomerID"] != "")]
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    if df.empty:
        raise ValueError("No valid transactions remain after cleaning. Check the input file and date/column formats.")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max().normalize()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    # High score is good: low recency gets high score; high frequency/monetary gets high score.
    rfm["R_Score"] = 6 - safe_quantile_score(rfm["Recency"], ascending=True)
    rfm["F_Score"] = safe_quantile_score(rfm["Frequency"], ascending=True)
    rfm["M_Score"] = safe_quantile_score(rfm["Monetary"], ascending=True)
    rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].astype(str).agg("".join, axis=1)
    rfm["RFM_Total"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm.sort_values(["RFM_Total","Monetary"], ascending=False), reference_date

def assign_segment(row):
    r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f >= 2:
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "Recent Customers"
    if r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    if r <= 2 and f >= 4:
        return "Can't Lose Them"
    if r <= 2 and f <= 2:
        return "Hibernating / Lost"
    return "Needs Attention"

def save_outputs(rfm, reference_date, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(out_dir / "rfm_customer_segments.csv", index=False)
    summary = rfm.groupby("Segment").agg(
        Customer_Count=("CustomerID","nunique"),
        Total_Revenue=("Monetary","sum"),
        Average_Recency_Days=("Recency","mean"),
        Average_Orders=("Frequency","mean"),
        Average_Customer_Spend=("Monetary","mean")
    ).reset_index().sort_values("Total_Revenue", ascending=False)
    summary.to_csv(out_dir / "segment_summary.csv", index=False)
    counts = rfm["Segment"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(10,6))
    counts.plot(kind="barh")
    plt.title("Customer Count by RFM Segment")
    plt.xlabel("Number of customers")
    plt.ylabel("Customer segment")
    plt.tight_layout()
    plt.savefig(out_dir / "rfm_segment_distribution.png", dpi=160)
    plt.close()
    revenue = summary.sort_values("Total_Revenue", ascending=True).set_index("Segment")["Total_Revenue"]
    plt.figure(figsize=(10,6))
    revenue.plot(kind="barh")
    plt.title("Revenue Contribution by RFM Segment")
    plt.xlabel("Total revenue")
    plt.ylabel("Customer segment")
    plt.tight_layout()
    plt.savefig(out_dir / "revenue_by_segment.png", dpi=160)
    plt.close()
    print(f"Reference date: {reference_date.date()}")
    print(f"Customers analysed: {len(rfm):,}")
    print(f"Outputs saved to: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run RFM customer segmentation.")
    parser.add_argument("--input", default=str(ROOT/"data/sample_online_retail.csv"),
                        help="Path to transaction CSV. Defaults to the included demo sample.")
    parser.add_argument("--output-dir", default=str(ROOT/"outputs"), help="Output directory.")
    args = parser.parse_args()
    rfm, reference_date = build_rfm(Path(args.input))
    save_outputs(rfm, reference_date, Path(args.output_dir))

if __name__ == "__main__":
    main()
