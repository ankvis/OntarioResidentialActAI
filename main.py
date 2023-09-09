import streamlit as st
import pinecone
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


def get_existing_vector():
    index_name = pinecone.list_indexes()[0]
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name,embeddings)

    return vectorstore



def ques_answer(vector_store, query):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search='similarity', search_kwargs = {'k' : 5})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(query)

    return answer

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        del st.session_state['query']

# def reset_page():
#     if "query" in st.session_state:
#         del st.session_state['query']
        
        

if __name__ == "__main__":
    import os
    import time
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))

    st.title('Ontario Residential Tenacies Act :derelict_house_building:')
    #st.image('ontario.jpg', use_column_width='always', caption ='Ontario')
    #st.divider()
    
    
    query = st.text_input('What would you like to know about the residential act?', key='query')
    #st.session_state.query
    # st.button(label="Reset",type='primary', on_click=reset_page)  
    
    if query:
        answer = ques_answer(get_existing_vector(), query)
        st.text_area('LLM Answer :sunglasses:', value=answer, height = 200)
    
        
        st.divider()
    
        if 'history' not in st.session_state:
            st.session_state.history = ''
        
        value = f'Q: {query} \nA: {answer}'
        
        st.session_state.history = f'{value} \n {"-"*100} \n {st.session_state.history}' 
        h = st.session_state.history
        
        st.text_area(label='Chat History', value=h, key='history', height=400)
        st.button("Clear history", type="primary", on_click=clear_history)
        #st.write(st.session_state['query']) 

        