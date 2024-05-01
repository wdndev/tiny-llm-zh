import json
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


st.set_page_config(page_title="Tiny LLM 92M Demo")
st.title("Tiny LLM 92M Demo")

model_id = "outputs/ckpt/tiny_llm_sft_92m"

@st.cache_resource
def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_id)
    return model, tokenizer, generation_config


def clear_chat_messages():
    del st.session_state.messages


def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是由wdndev开发的个人助手，很高兴为您服务😄")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []
    
    return st.session_state.messages


max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 1024, 512, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 0, step=1)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 1.0, step=0.01)
do_sample = st.sidebar.checkbox("do_sample", value=True)

def main():
    model, tokenizer, generation_config = load_model_tokenizer()
    messages = init_chat_messages()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()

            generation_config.max_new_tokens = max_new_tokens
            generation_config.top_p = top_p
            generation_config.top_k = top_k
            generation_config.temperature = temperature
            generation_config.do_sample = do_sample
            print("generation_config: ", generation_config)

            sys_text = "你是由wdndev开发的个人助手。"
            messages.append({"role": "user", "content": prompt})
            user_text = prompt
            input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                                    "<|user|>", user_text.strip(), 
                                    "<|assistant|>"]).strip() + "\n"

            model_inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(model_inputs.input_ids, generation_config=generation_config)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            placeholder.markdown(response)
        
        messages.append({"role": "assistant", "content": response})
        print("messages: ", json.dumps(response, ensure_ascii=False), flush=True)

    st.button("清空对话", on_click=clear_chat_messages)


if __name__ == "__main__":
    main()