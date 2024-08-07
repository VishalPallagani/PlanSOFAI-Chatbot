import random

import ollama
import streamlit as st

st.title("Plan-SOFAI Chatbot")

# Initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize model selection
if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models)


# Define a function to preprocess the user input
def preprocess_input(user_input):
    # Will be adding a relevant example from chromadb (in the future)
    user_input = """
    ### Role: You are a state-of-the-art automated planner.
    
    ### Instruction: You are required to output the plan for a given planning task.
    
    ### Task: 
     
    I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the, actions I can do:
    
    - Pick up a block
    - Unstack a block from on top of another block
    - Put down a block
    - Stack a block on top of another block
    
    I have the following restrictions on my actions:
    
    - I can only pick up or unstack one block at a time.
    - I can only pick up or unstack a block if my hand is empty.
    - I can only pick up a block if the block is on the table and the block is clear. A block is clear, if the block has no other blocks on top of it and if the block is not picked up.
    - I can only unstack a block from on top of another block if the block I am unstacking was really on, top of the other block.
    - I can only unstack a block from on top of another block if the block I am unstacking is clear.
    - Once I pick up or unstack a block, I am holding the block.
    - I can only put down a block that I am holding.
    - I can only stack a block on top of another block if I am holding the block being stacked.
    - I can only stack a block on top of another block if the block onto which I am stacking the block, is clear.
    - Once I put down or stack a block, my hand becomes empty.
    - Once you stack a block on top of a second block, the second block is no longer clear.
    
    [STATEMENT]
    
    As initial conditions I have that, the red block is clear, the blue block is clear, the yellow 
    block is clear, the hand is empty, the blue block is on top of the orange block, the red block 
    is on the table, the orange block is on the table and the yellow block is on the table.

    My goal is to have that the orange block is on top of the blue block.
    
    My plan is as follows:
    
    [PLAN]
    unstack the blue block from on top of the orange block
    put down the blue block
    pick up the orange block
    stack the orange block on top of the blue block
    [PLAN END]
    
    [STATEMENT]
    
    As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is 
    empty, the red block is on top of the blue block, the yellow block is on top of the orange 
    block, the blue block is on the table and the orange block is on the table.

    My goal is to have that the orange block is on top of the red block.
    
    My plan is as follows:
    
    [PLAN]"""
    return f"{user_input}"


# Define an evaluation function
def evaluate_response(response):
    # Implement your evaluation logic here
    # Return True if the plan is correct, False otherwise
    return random.choice([True, False])


# Define a function to generate a response from the model
def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If user inputs a message
if prompt := st.chat_input("What planning problem are we solving?"):
    # Preprocess the user's input
    modified_input = preprocess_input(prompt)

    # Add the modified input to the history
    st.session_state["messages"].append({"role": "user", "content": modified_input})

    # Display the modified input
    with st.chat_message("user"):
        st.markdown(modified_input)

    # Initialize a variable to track if the plan is correct
    plan_correct = False
    iteration = 0
    max_iterations = 5

    # Loop to generate responses and evaluate them
    while not plan_correct and iteration < max_iterations:
        iteration += 1

        with st.chat_message("assistant"):
            # Generate a response from the model
            response = st.write_stream(model_res_generator())
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )

        # Evaluate the response
        plan_correct = evaluate_response(response)

        if plan_correct:
            st.success("The above plan is correct!")
        else:
            feedback = "Please provide a correct plan."
            st.error(f"The above plan is not correct. Providing feedback :{feedback}")
            st.error("Generating a new plan ..")
            st.session_state["messages"].append({"role": "user", "content": feedback})

    if not plan_correct:
        st.warning("Reached maximum iterations without a correct plan.")
