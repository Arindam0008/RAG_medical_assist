
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
# Walrus operator (:=) assigns input to user_query if provided
if user_query := st.chat_input("Ask me something..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve docs + generate response
    retrieved_docs = retriever.get_relevant_documents(user_query)
    # retriever.get_relevant_documents(user_query) returns a list of Document objects, not plain text.
    # we should join their contents
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = generate_prompt(retrieved_context, user_query)
    ai_output = generate_medical_advice(prompt)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": ai_output})

    # Show/Render assistant message
    with st.chat_message("assistant"):
        st.markdown(ai_output)

