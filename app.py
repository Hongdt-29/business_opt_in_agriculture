import streamlit as st
from rasterio.io import MemoryFile
import rasterio
import numpy as np
import pandas as pd
import joblib
import math
import os
import rasterio.mask

from streamlit_folium import folium_static
import folium
import streamlit.components.v1 as components
import base64

import rasterio
import rasterio.features
import rasterio.warp

#Page size config
st.set_page_config(
            page_title="Sky Crop",
            page_icon="🌾",
            layout="wide")

CSS = """
    iframe {
        width: auto;
        height: 500px;
    }
    img {
        object-fit: cover;
        height: 500px;
    }
    .logo-img {
        object-fit: cover;
        height: 300px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton button {
        font-size: 24px;
        text-align: center;

    }
    .e16nr0p30 p {
        font-size: 24px;
    }

    """


st.write(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.title('Rice and Sugarcane Classifier 🌾')
def main():
    file_uploaded = st.file_uploader("", type=["tif"])

    if file_uploaded is not None:
        with MemoryFile(file_uploaded) as memfile:
            with memfile.open() as img:
                geo_json = get_geo_json(img)
                pre_image = image_preprocessing(img,file_uploaded)
                # st.write(pre_image)

                #Display Location map and satellite mage
                col1, col2 = st.beta_columns(2)
                col1.header("Location 📍 ")
                with col1:
                    display_map(geo_json)
                col2.header("Satellite Image 🛰️ ")
                uploadedFileName = file_uploaded.name
                uploadedFileName = uploadedFileName.replace('.tif','')
                col2.image(f'raw_data/demo/satellite/{uploadedFileName} (red borders).jpg',use_column_width=True)

                # Question Face
                st.markdown("<h2 style='text-align: left; color: white;'>Rice or Sugarcane ? 🤔 </h2>", unsafe_allow_html=True)
                st.text("")
                #Predict
                class_btn = st.button("Classify")
                if class_btn:
                    with st.spinner('Model working....'):
                        prediction = predict(pre_image)[0]
                        if prediction == 'rice_1y':
                            prediction = 'Rice'
                            st.markdown(f"""<img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('Rice.png', "rb").read()).decode()}">
                                        """,unsafe_allow_html=True)
                            st.text("")
                            st.text("")
                            st.success("Predicted Class: " + prediction)
                        else:
                            st.markdown(f"""<img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('Sugarcane.png', "rb").read()).decode()}">
                                        """,unsafe_allow_html=True)
                            st.text("")
                            st.text("")
                            st.success("Predicted Class: " + prediction)


                img.close()

def image_preprocessing(img,file_uploaded):
    X = []
    filename = file_uploaded.name
    file_id, date = filename.split('_all_bands_')
    date = date.replace('.tif','')


    # Compute ndvi mean, median, std
    ndvi = (img.read(8)-img.read(4))/(img.read(8) + img.read(4))
    ndvi_ = np.nan_to_num(ndvi, nan=-1)
    ndvi_values = np.array([x for x in ndvi_.flatten() if x != -1])
    ndvi_mean = ndvi_values.mean()
    ndvi_median = np.median(ndvi_values)
    ndvi_std = np.std(ndvi_values)

    # Compute mi mean, median, std
    mi = (img.read(8)-img.read(11))/(img.read(8) + img.read(11))
    mi_ = np.nan_to_num(mi, nan=-1)
    mi_values =np.array([x for x in mi_.flatten() if x != -1])
    mi_mean = mi_values.mean()
    mi_median = np.median(mi_values)
    mi_std = np.std(mi_values)

    # Compute Cb1 mean, median, std
    band_combo_1 = img.read(7) + img.read(6) + img.read(4)
    bc1_ = np.nan_to_num(band_combo_1, nan=-1)
    bc1_values =np.array([x for x in bc1_.flatten() if x != -1])
    bc1_mean = bc1_values.mean()
    bc1_median = np.median(bc1_values)
    bc1_std = np.std(bc1_values)

    # Compute Cb2 mean, median, std
    band_combo_2 = img.read(11) + img.read(8) + img.read(2)
    bc2_ = np.nan_to_num(band_combo_2, nan=-1)
    bc2_values =np.array([x for x in bc2_.flatten() if x != -1])
    bc2_mean= bc2_values.mean()
    bc2_median= np.median(bc2_values)
    bc2_std= np.std(bc2_values)

    # Compute Cb3 mean, median, std
    band_combo_3 = (img.read(3)-img.read(8))/(img.read(3)+img.read(8))
    bc3_ = np.nan_to_num(band_combo_3, nan=-1)
    bc3_values =np.array([x for x in bc3_.flatten() if x != -1])
    bc3_mean= bc3_values.mean()
    bc3_median= np.median(bc3_values)
    bc3_std= np.std(bc3_values)

    # Compute Cb4 mean, median, std
    band_combo_4 = img.read(12) + img.read(8) + img.read(4)
    bc4_ = np.nan_to_num(band_combo_4, nan=-1)
    bc4_values =np.array([x for x in bc4_.flatten() if x != -1])
    bc4_mean= bc4_values.mean()
    bc4_median= np.median(bc4_values)
    bc4_std= np.std(bc4_values)

    # Compute Cb5 mean, median, std
    band_combo_5 = img.read(4) + img.read(3) + img.read(2)
    bc5_ = np.nan_to_num(band_combo_5, nan=-1)
    bc5_values =np.array([x for x in bc5_.flatten() if x != -1])
    bc5_mean= bc5_values.mean()
    bc5_median= np.median(bc5_values)
    bc5_std= np.std(bc5_values)

    b_dict = {'date':date,\
              'ndvi_mean': ndvi_mean,'ndvi_median': ndvi_median,'ndvi_std': ndvi_std,\
              'mi_mean': mi_mean,'mi_median': mi_median,'mi_std': mi_std,\
              'bc1_mean': bc1_mean,'bc1_median': bc1_median,'bc1_std': bc1_std,\
              'bc2_mean': bc2_mean,'bc2_median': bc2_median,'bc2_std': bc2_std,\
              'bc3_mean': bc3_mean,'bc3_median': bc3_median,'bc3_std': bc3_std
              # 'bc4_mean': bc4_mean,'bc4_median': bc4_median,'bc4_std': bc4_std,\
              # 'bc5_mean': bc5_mean,'bc5_median': bc5_median,'bc5_std': bc5_std
            }

    for band in range(1,14):
        b_dict[f'b{band}_mean'] = img.read(band).flatten().mean()
        b_dict[f'b{band}_median'] = np.median(img.read(band).flatten())
        b_dict[f'b{band}_std'] = np.std(img.read(band).flatten())

    X.append(b_dict)
    df = pd.DataFrame(X)
    df['month'] = pd.DatetimeIndex(df['date']).month
    df = df.drop(columns = 'date')

    return df


def predict(image):
    # pipeline_model = "pipeline.joblib"
    # model = joblib.load(pipeline_model)
    pipeline_test_model = "pipeline_final95.joblib"
    model = joblib.load(pipeline_test_model)

    predictions = model.predict(image)

    return predictions


def get_geo_json(dataset):
    # with rasterio.open('./sugarcane/Name_5f2a53d4868954001c94d20b_all_bands_2020-07-08.tif') as dataset:
    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()
    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):
        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, 'EPSG:4326', geom, precision=6)
    return geom

def display_map(geo_json):
    longitude = geo_json['coordinates'][0][0][0]
    latitude = geo_json['coordinates'][0][0][1]

    m = folium.Map(location=[latitude, longitude], zoom_start= 6)

    folium.Marker(
        location=[latitude, longitude],
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.GeoJson(geo_json).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)

if __name__ == "__main__":
    main()

