import streamlit as st
import pandas as pd
from os.path import join, dirname
from PIL import Image
from utils import loader
from utils import inspector as inspector

def main():
    script_dir = dirname(__file__)
    st.image(join(script_dir, "../ressources/banner2.png"), use_column_width=True)

    st.sidebar.title("Inspect")
    model_type = st.sidebar.selectbox("Select Model", ["AutoEncoder", "PCA", "ZScore"])

    st.sidebar.subheader("Upload Data")
    up_demog = st.sidebar.file_uploader("Demographics (.csv)", type="csv")
    up_data = st.sidebar.file_uploader("Tract Profiles (.xlsx)", type="xlsx")

    if not up_demog or not up_data:
        st.warning("Upload both files to begin.")
        return

    df_demog = pd.read_csv(up_demog)
    datasheet = loader.load_data(up_data)

    metric = st.sidebar.selectbox("Metric", list(datasheet.keys()))
    df_data = loader.combine_demog_and_data(df_demog, datasheet, metric)

    subject = st.sidebar.selectbox("Subject", df_demog.ID)

    tract_list = sorted(set(col.split("_")[0] for col in df_data.columns if "_" in col))
    selected_tracts = st.sidebar.multiselect("Select Tracts", tract_list, default=tract_list)

    regress = st.sidebar.checkbox("Regress Confounds", value=True)

    title = st.sidebar.text_input("Savename", "MY_ANALYSIS")

    if st.sidebar.button("Run Analysis"):
        x, x_hat, bin_vector, global_score, sid, y_test = inspector.run(
            subject, df_data, df_demog, regress, selected_tracts, metric, model_type, title
        )

        if x is None:
            st.error("Model could not run. Check tract selection or data format.")
            return

        st.subheader("Results for Subject " + str(sid))
        st.write("Global Anomaly Score:", global_score)

        st.subheader("Tract-Level Anomalies")
        st.write(pd.DataFrame(bin_vector.reshape(1, -1), columns=x.columns))

if __name__ == "__main__":
    main()
