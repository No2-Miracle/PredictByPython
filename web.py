import streamlit as st
from PIL import Image
from predictInWeb import predict


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("欢迎使用斜视识别 App")
st.write("")
st.write("")


file_up = st.file_uploader("请上传正脸照片", type="jpg")
if file_up is None:
    st.write("等待上传")

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("正在分析，请稍等...")
    labels = predict(file_up)
    st.success('识别成功')
    st.write("")
    st.write("识别结果为: " + labels)

