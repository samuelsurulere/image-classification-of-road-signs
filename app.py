import streamlit as st
import utils
import base64
import warnings

warnings.filterwarnings("ignore")


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


st.title("Traffic road signs classification model") 
st.markdown('<h2 style="color:white;"> This vision classification model classifies uploaded images into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h5 style="color:white;"> bump, construction, divided-highway-begins, divider-ahead, handicap, no-parking-either-sides,\
            pedestrian-crossing, people-working, railway-crossing, shared-use, speed-limit-80, stop, yield </h3>', unsafe_allow_html=True)


st.sidebar.write("""
This project is an end-to-end **CNN Image Classification Model** based on identifying the traffic road signs in Alberta province.
Due to lack of dataset for this specific problem, one shot learning was used. Transfer learning was employed to train the CNN models applied.
Other models were also used to compare performance and accuracy. The EfficientNetB3 model came out on top and had smaller size compared to other models.
The methods used were as a result of the complexity of the defined problem and lack of data. The model was trained on a dataset of 13 classes of road signs.
After training and retraining the model for extensive hours, the following benchmarks were achieved:

**Accuracy:** **`86%`**

**Architecture:** **`EfficientNetB3`**

**Trainable Parameters:** **`11,752,501`**

**Parameters Size:** **`44.83MB`**
"""
)

model_path = 'output/model_temp.pth'

classes = ('bump','construction','divided-highway-begins',
            'divider-ahead','handicap',
            'no-parking-either-sides',
            'pedestrian-crossing','people-working',
            'railway-crossing','shared-use',
            'speed-limit-80','stop','yield')


def main():
    uploaded_file = st.file_uploader(label='Insert image for classification', type=['png'])
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
