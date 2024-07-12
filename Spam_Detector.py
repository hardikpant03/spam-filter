import pickle
import streamlit as st
from gtts import gTTS
import pygame
 
pygame.mixer.init()

def text_to_speech(Text):
    language = "en"
    tts = gTTS(text=Text, lang=language, slow=False)
    print(tts)
    # Save the Audio file to a temporary file
    tts.save("output.mp3")
    
    pygame.mixer.music.load("output.mp3")

    pygame.mixer.music.play()

model = pickle.load(open("spam.pkl" , "rb"))
cv = pickle.load(open("vectorizer.pkl" , "rb"))

def main():
    st.title("Email Spam Classification Apps")
    st.subheader("Build with Streamlit & Python")
    msg=st.text_input("Enter the text: ")

    if st.button("Predict"):
        data=[msg]
        vect=cv.transform(data).toarray()
        pred = model.predict(vect)
        result = pred[0]
        print(result)
        if result == 1:
            st.error("This is a spam mail")
            text_to_speech("This is a spam mail")
        else:
            st.success("This is Ham mail")
            text_to_speech("This is a ham mail")
    footer = """
    <hr>
    <p style="text-align:center;">Made by Hardik and Yash</p>
    """

    st.markdown(footer, unsafe_allow_html=True)
        

main()