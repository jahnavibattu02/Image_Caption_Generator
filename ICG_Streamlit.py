import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np # numpy - expand_dims 
from streamlit_lottie import st_lottie
import streamlit as st
import requests
import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import load_model

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def main():
    
    st.set_page_config(layout="wide")

    option_type = option_menu(
        menu_title = None,
        options = ["Home","Work","Contact"],
        icons = ["house","gear","envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
    )
    
    if option_type == "Home":

        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Image Caption Generator</h1>",unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_1GkVPfS0cv.json")

        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="high", # medium ; high
            #renderer="svg", # canvas
            height=400,
            width=-900,
            key=None,
        )        
        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Upload your image and see the magic of Neural Network.</h4>",unsafe_allow_html=True)
    elif option_type == "Work":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Please Upload your Image</h1>",unsafe_allow_html=True)
        img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if img is not None:
            st.image(img)
            st.markdown("<h1 style='text-align: center; color: #08e08a;'>Caption</h1>",unsafe_allow_html=True)
            image = load_img(img.name, target_size=(224, 224))
            # convert image pixels to numpy array
            image = img_to_array(image)
            # reshape data for model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # preprocess image for vgg
            image = preprocess_input(image)
            # extract features
            feature = model.predict(image, verbose=0)
            with open('./tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)

            model1 = load_model('./best_model.h5')

            in_text = 'startseq'
            # iterate over the max length of sequence
            for i in range(35):
                # encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad the sequence
                sequence = pad_sequences([sequence], 35)
                # predict next word
                yhat = model1.predict([feature, sequence], verbose=0)
                # get index with high probability
                yhat = np.argmax(yhat)
                # convert index to word
                word = idx_to_word(yhat, tokenizer)
                # stop if word not found
                if word is None:
                    break
                # append word as input for generating next word
                in_text += " " + word
                # stop if we reach end tag
                if word == 'endseq':
                    break
            st.markdown("<h3 style='text-align: center; color: #dae5e0;'>"+in_text[9:-6]+"</h3>",unsafe_allow_html=True)


    else:
            st.subheader("About")
            st.text("This is a Virtual Notes Assistant, Made by a students of the IIIT-Kalyani, India.")
            st.text("Jahnavi Pravaleeka Battu")
            st.text("Sai Dileep Kumar Mukkamala")
            st.text("Senior CSE Students @ IIITKalyani")
            st.text("Data Science Enthusiasists.")

            lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_in4cufsz.json")

            st_lottie(
                lottie_hello,
                speed=1,
                reverse=False,
                loop=True,
                quality="high", # medium ; high
                #renderer="svg", # canvas
                height=300,
                width=-900,
                key=None,
            )

        
            st.header(":mailbox: Get In Touch With Us!")


            contact_form = """
            <form action="https://formsubmit.co/msaidileepkumar2002@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder="Your message here"></textarea>
                <button type="submit">Send</button>
            </form>
            """

            st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
            def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


            local_css("style/style.css")



if __name__ == '__main__':
    main()