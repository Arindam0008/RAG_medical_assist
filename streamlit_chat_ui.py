
import streamlit as st
from rag_medical_query import retriever, generate_prompt, generate_medical_advice

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
st.title("🧠 AI Medical Assistant")
user_query = st.text_input('Enter the query')
if st.button("Enquire"):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve docs + generate response
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = generate_prompt(context, user_query)
    ai_output = generate_medical_advice(prompt)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": ai_output})

    # Show/Render assistant message
    with st.chat_message("assistant"):
        st.markdown(ai_output)

