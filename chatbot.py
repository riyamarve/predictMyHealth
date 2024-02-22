from openai import OpenAI
import streamlit as st

st.title("HEALTH CHATBOT")

client = OpenAI(api_key="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")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to display messages
def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        # Generate a short and descriptive prompt for the marketing poster
        poster_prompt = (
            "Create a funny attractive cartoon to enlighten the mood of the viewer. "
            "Highlight attractive cartoons, funny jokes, bright colours and themes "
            "beautifully displayed cartoons, inviting visuals."
            "Convey a sense of humour and light-mindedness. Size: 1200x800."
        )

        # Attempt to display the generated image
        try:
            image_response = client.images.generate(prompt=poster_prompt, n=1, size="1024x1024")
            image_url = image_response.data[0].url
            st.image(image_url, caption="Generated Marketing Poster", use_column_width=True)
            st.session_state.messages.append({"role": "assistant", "content": "Cartoons generated"})
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

# Display all messages
display_messages()
