import streamlit as st
import utils
import base64
import warnings
import torch


warnings.filterwarnings("ignore")

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(f"""<style>.stApp {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: cover
        }}
        </style>""", unsafe_allow_html=True)

set_bg_hack('figures/background.png')

def sidebar_bg(side_bg):
    side_bg_ext = 'png'
    st.markdown(f"""<style>[data-testid="stSidebar"] > 
                div:first-child {{
                    background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
                    }}
                    </style>""", unsafe_allow_html=True,
                    )

sidebar_bg('figures/sidebar.png')


st.title("Traffic road signs classification model for Alberta province") 
st.markdown('<h2 style="color:white;"> This vision model classifies uploaded images into one of the following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h5 style="color:white;"> \
            animal crossing, bicycle lane, bump, construction, divided highway begins, \
            handicap, keep right, no entry, no parking on either sides,\
            no u-turn, pedestrian crossing, people working, playground zone, \
            railway crossing, 80 km/h speed limit, stop, yield </h3>', unsafe_allow_html=True)


st.sidebar.write("""
This project is an end-to-end **CNN Image Classification Model** based on identifying the traffic road signs in Alberta province.
Due to lack of dataset for this specific problem, one shot learning was used. Transfer learning was employed to train three CNN architectures 
(EfficientNetB3, VGG16 and RestNet50). The EfficientNetB3 model came out on top and was lightweight compared to the other architectures.
The model was trained on a dataset of 17 classes of road signs. After extensively training, retraining and hyperparameter tuning of the EfficientNetB3 architecture 
for extensive hours, the following benchmarks were achieved:

**Accuracy:** **`86%`**

**Trainable Parameters:** **`11,752,501`**

**Parameters Size:** **`44.84 MB`**
"""
)

model_path = 'output/model_temp.pth'

classes = (
        'animal-crossing','bicycle-lane','bump',
        'construction','divided-highway-begins','handicap',
        'keep-right','no-entry','no-parking-either-side',
        'no-u-turn','pedestrian-crossing','people-working',
        'playground-zone','railway-crossing','shared-use',
        'speed-limit-80','stop','yield'
    )


def main():
    uploaded_file = st.file_uploader(label='Insert image for classification', type=['jpg','png','jpeg'])
    if uploaded_file:
        image = utils.load_image(uploaded_file, device)
        model = utils.load_model(model_path, device)
        result = st.button('Predict the name of this road sign')
        if result:
            st.write('Right back at you in a jiffy...')
            utils.predict(image, model, classes)
    else:
        return None


if __name__ == '__main__':
    main()