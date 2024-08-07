import re
import subprocess

import ollama
import streamlit as st

st.title("Plan-SOFAI Chatbot")

# Initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize model selection
if "model" not in st.session_state:
    st.session_state["model"] = ""

# Fetch available models
models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models)


# Function to preprocess the user input
def preprocess_input(domain_content, problem_content):
    # Prepare the prompt based on the domain and problem files
    user_input = f"""
    PDDL Domain Definition
    
    {domain_content}

    PDDL Problem Definition
    {problem_content}
    
    Task
    
    Generate a plan to achieve the goal state from the initial state using the given domain and problem definitions.
    ONLY OUTPUT THE PLAN WITH EACH STEP ENCLOSED IN A PARANTHESIS AND NOTHING ELSE.
    """
    return user_input


def process_plan(response):
    # keep all the text in the predicate format, and place it in the order it appears in the text.
    # remove all the text that is not in the predicate format.

    re_code = re.compile(r"\(.*?\)")
    plan = re_code.findall(response)

    # make sure the predicate only has one set of paranthesis
    for i in range(len(plan)):
        if plan[i].startswith("((") and plan[i].endswith("))"):
            plan[i] = plan[i][1:-1]
        elif plan[i].startswith("(("):
            plan[i] = plan[i][1:]
        elif plan[i].endswith("))"):
            plan[i] = plan[i][:-1]
    # plan = [p[1:-1] if p.startswith('((') and p.endswith('))') else p for p in plan]

    # print(plan)
    # raise KeyboardInterrupt

    return plan


# Evaluation function that calls the plan validator script
def evaluate_response(response, domain_content, problem_content):  # to be changed
    # Path to the validator
    validator_path = "./VAL/validate"

    # Write the domain, problem, and plan to temporary files
    with open("domain.pddl", "w") as domain_file:
        domain_file.write(domain_content)

    with open("problem.pddl", "w") as problem_file:
        problem_file.write(problem_content)

    plan_text = process_plan(response)
    with open("plan.txt", "w") as plan_file:
        plan_file.write("\n".join(plan_text))

    if plan_text == []:
        return False, "Plan is invalid. Reason: No plan found."

    try:
        # Run the validator command
        result = subprocess.run(
            [validator_path, "-v", "domain.pddl", "problem.pddl", "plan.txt"],
            capture_output=True,
            text=True,
        )

        # Store the entire output from running the command
        validator_output = result.stdout
        # print("validator_output: ", validator_output)
        # raise KeyboardInterrupt

        # Display the validator output for debugging
        # st.text("Validator Output:")
        # st.text(validator_output)

        # Check if the plan is valid based on the validator output
        if "plan valid " in validator_output.lower():
            return True, "Plan is valid"

        # Initialize variables
        failed_time = "Unknown"
        failure_reason = "Unknown reason"
        repair_advice = "No repair advice found"

        # Split the output into lines for easier processing
        lines = validator_output.splitlines()

        # Iterate through lines to find the failure time and reason
        for i, line in enumerate(lines):
            time_match = re.match(r"Checking next happening \(time (\d+)\)", line)
            if time_match and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("Plan failed because of"):
                    failed_time = time_match.group(1)
                    break

        # Iterate through lines to find the failure reason
        for i, line in enumerate(lines):
            if line.strip().startswith("Plan failed because of"):
                failure_reason = line.strip()
                # Continue concatenating lines until an empty line is encountered
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line == "":
                        break
                    failure_reason += " " + next_line
                break
        # Extract Plan Repair Advice if it exists
        repair_advice_match = re.search(
            r"Plan Repair Advice:(.*?)(Failed plans:|$)", validator_output, re.DOTALL
        )
        if repair_advice_match:
            # repair_advice = (
            #     repair_advice_match.group(1).strip().replace("at time", "at step")
            # )
            repair_advice = repair_advice_match.group(1).strip()
            # Remove 'at time X' part from repair advice
            repair_advice = re.sub(r"at time \d+", "", repair_advice)

            if failed_time == "Unknown" and failure_reason == "Unknown reason":
                return (
                    False,
                    f"Plan is invalid. Repair advice: {repair_advice}. Generate a correct plan for the given planning problem.",
                )

            elif failed_time == "Unknown" and failure_reason != "Unknown reason":
                return (
                    False,
                    f"Plan failed because of {failure_reason}. Repair advice: {repair_advice}. Generate a correct plan for the given planning problem.",
                )

            elif failed_time != "Unknown" and failure_reason == "Unknown reason":
                return (
                    False,
                    f"Plan failed at step {failed_time}. Repair advice: {repair_advice}. Generate a correct plan for the given planning problem.",
                )

            return (
                False,
                f"Plan failed at step {failed_time}.\n Failure reason: {failure_reason}. Generate a correct plan for the given planning problem.",
            )

        else:
            # identify all the text after Checking plan: plan.txt
            all_text = validator_output.split("Checking plan: plan.txt")[1]

            return False, f"Plan is invalid. Reason: {all_text}"

        # Default feedback if no repair advice is found
        return False, "Plan is invalid"

    except Exception as e:
        st.error(f"Error running plan validator: {e}")
        return False, str(e)


# Function to generate a response from the model
def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    response = ""
    for chunk in stream:
        response += chunk["message"]["content"]
    return response


# Function to run FastDownward as a fallback solver
def run_fastdownward(domain_content, problem_content):
    # Write the domain and problem content to temporary files
    with open("domain.pddl", "w") as domain_file:
        domain_file.write(domain_content)

    with open("problem.pddl", "w") as problem_file:
        problem_file.write(problem_content)

    try:
        result = subprocess.run(
            [
                "./downward/fast-downward.py",
                "domain.pddl",
                "problem.pddl",
                "--search",
                "astar(lmcut())",
            ],
            capture_output=True,
            text=True,
        )

        # Extract the plan from FastDownward's output
        plan_match = re.search(
            r"Solution found.*?\n(.*?)(?:\n\n|\Z)", result.stdout, re.DOTALL
        )
        if plan_match:
            # Use regex to find all the plan steps
            plan_steps = re.findall(
                r"(?:unstack|put-down|pick-up|stack) \w+.*", plan_match.group(1)
            )
            formatted_plan = "\n".join([f"({step.strip()})" for step in plan_steps])
            return formatted_plan
        return "No solution found by FastDownward."
    except Exception as e:
        return f"Error running FastDownward: {e}"


# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File upload for domain and problem files
domain_file = st.file_uploader("Upload Domain File", type=["txt", "pddl"])
problem_file = st.file_uploader("Upload Problem File", type=["txt", "pddl"])

if domain_file and problem_file:
    domain_content = domain_file.read().decode("utf-8")
    problem_content = problem_file.read().decode("utf-8")

    # Preprocess the user input
    modified_input = preprocess_input(domain_content, problem_content)

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
            response = model_res_generator()
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
            st.markdown(response)

        # Evaluate the response using the plan validator
        plan_correct, feedback = evaluate_response(
            response, domain_content, problem_content
        )

        if plan_correct:
            st.success("The above plan is correct!")
        else:
            st.error(f"The above plan is not correct. Providing feedback: {feedback}")
            st.error("Generating a new plan ...")
            # Use feedback as the new user prompt
            st.session_state["messages"].append({"role": "user", "content": feedback})

    if not plan_correct:
        st.warning("Reached maximum iterations without a correct plan.")
        st.info(
            "System 1 solver could not solve the planning problem in 5 turns, so invoking System 2 solver."
        )
        fastdownward_plan = run_fastdownward(domain_content, problem_content)
        st.markdown(f"#### Plan generated by FastDownward:\n{fastdownward_plan}")
