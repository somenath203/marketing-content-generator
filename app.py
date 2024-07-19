import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')

llm_model = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.7, model_kwargs={'max_length': 128, 'token': os.getenv('HF_TOKEN')})


st.set_page_config(
    page_title='Marketing Content Generator',
    page_icon='✅',
    layout='centered',
    initial_sidebar_state='collapsed'
)


st.header('Marketing Content Generator')


user_product_name_input = st.text_input('Enter the name of the product')

user_product_description_input = st.text_area('Enter the description of the product', height=275)


tasktype_option = st.selectbox(
    'Which type of task you want to be performed',
    ('Write a sales copy', 'Write a tweet', 'Write a product description')
)


submit = st.button('Generate Content')


if submit:

    if user_product_name_input and user_product_description_input:

        prompt_template = PromptTemplate(
            template="""
            You are a helpful assistant. Perform the task according to the type: {tasktype_option} on the name of 
            of the product: {user_product_name_input} and the description of the product: {user_product_description_input}. Generate to the point answer, not any useless information and if needed, use emojis wherever required. 
            And atlast, tell user this thing like feel free to modify the {tasktype_option} according to your need".
            And last thing, don't generate same information multiple times. 
            """,
            input_variables=["tasktype_option", "user_product_name_input", "user_product_description_input"]
        )

        with st.spinner('Generating Content...'):

            response = llm_model.invoke(prompt_template.format(tasktype_option=tasktype_option, user_product_name_input=user_product_name_input, user_product_description_input=user_product_description_input))

            st.success('Content generated successfully')

            st.markdown(response)

    else:

        st.error('Please fill both the input fields')