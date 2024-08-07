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


def modify_pddl_file(flag):
    if not flag:
        return

    input_file = "problem.pddl"
    output_file = "lifted_problem.pddl"

    with open(input_file, "r") as file:
        lines = file.readlines()

    objects = []
    goal_section = []
    in_objects_section = False
    in_init_section = False
    in_goal_section = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("(:objects"):
            in_objects_section = True
        elif stripped_line.startswith("(:init"):
            in_objects_section = False
            in_init_section = True
        elif stripped_line.startswith("(:goal"):
            in_init_section = False
            in_goal_section = True

        if in_objects_section:
            objects.extend(stripped_line.split())

        if in_goal_section:
            goal_section.append(line)

    # Filter out non-block objects
    blocks = [obj for obj in objects if obj != "(:objects" and obj != ")"]

    # Create new init section
    new_init_section = ["(:init\n"]
    for block in blocks:
        new_init_section.append(f"    (ontable {block})\n")
        new_init_section.append(f"    (clear {block})\n")
    new_init_section.append("    (handempty)\n")
    new_init_section.append(")\n")

    # Write the new problem file
    with open(output_file, "w") as file:
        for line in lines:
            if line.strip().startswith("(:init"):
                file.writelines(new_init_section)
                in_init_section = True
            elif in_init_section and line.strip() == ")":
                in_init_section = False
            elif not in_init_section:
                file.write(line)


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


def process_plan(response):  # changepoint
    # keep all the text in the predicate format, and place it in the order it appears in the text.
    # remove all the text that is not in the predicate format.

    # Updated regex to capture predicates with potential missing closing parenthesis
    re_code = re.compile(r"\([^\(\)]*")
    plan = re_code.findall(response)

    # Debug print to check what the regex captured
    print("Captured predicates:", plan)

    # make sure the predicate only has one set of parentheses
    for i in range(len(plan)):
        plan[i] = plan[i].strip()
        # Check if the predicate is missing a closing parenthesis and add it if necessary
        if plan[i].count("(") > plan[i].count(")"):
            plan[i] += ")"

    # Filter out any empty or invalid predicates
    plan = [p for p in plan if p != "()" and p != ""]

    print("Processed Plan:", plan)

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
                    f"Plan is invalid. Repair advice: {repair_advice}.",
                )

            elif failed_time == "Unknown" and failure_reason != "Unknown reason":
                return (
                    False,
                    f"Plan failed because of {failure_reason}. Repair advice: {repair_advice}.",
                )

            elif failed_time != "Unknown" and failure_reason == "Unknown reason":
                return (
                    False,
                    f"Plan failed at step {failed_time}. Repair advice: {repair_advice}.",
                )

            return (
                False,
                f"Plan failed at step {failed_time}.\n Failure reason: {failure_reason}.",
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


# function to extract init and goal from the problem file
def extract_init_and_goal(lifted_problem_content):
    # Split the content into lines
    lines = lifted_problem_content.split("\n")

    # Initialize variables to store the init and goal sections
    init_section = []
    goal_section = []

    # Flags to track which section we are in
    in_init_section = False
    in_goal_section = False

    # Iterate through each line in the content
    for line in lines:
        stripped_line = line.strip()

        # Check if the line marks the beginning of the init section
        if stripped_line.startswith("(:init"):
            in_init_section = True
            init_section.append(line)
        # Check if the line marks the beginning of the goal section
        elif stripped_line.startswith("(:goal"):
            in_goal_section = True
            in_init_section = False
            goal_section.append(line)
        # Check if the line marks the end of a section
        elif stripped_line == ")":
            if in_init_section:
                init_section.append(line)
                in_init_section = False
            elif in_goal_section:
                goal_section.append(line)
                in_goal_section = False
        # Add lines to the respective sections
        elif in_init_section:
            init_section.append(line)
        elif in_goal_section:
            goal_section.append(line)

    # Join the lines to form the init and goal sections as strings
    init_section_str = "\n".join(init_section)
    goal_section_str = "\n".join(goal_section)

    return init_section_str, goal_section_str


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

    # Initialize a variable to store the previous failure reason
    previous_failure_reasons = list()
    feedback_example = False

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
            if not feedback_example:
                feedback_example = True
                modify_pddl_file(True)
                fastdownward_plan = run_fastdownward(
                    domain_content,
                    open("lifted_problem.pddl", "rb").read().decode("utf-8"),
                )
                # print(fastdownward_plan)
                lifted_problem_content = (
                    open("lifted_problem.pddl", "rb").read().decode("utf-8")
                )
                init_str, goal_str = extract_init_and_goal(lifted_problem_content)

                # update feedback with lifted problem content and the plan generated by FastDownward
                feedback = f"{feedback}\n\nFor example, if the initial conditions are {init_str} and the goal conditions are {goal_str}, the correct plan is {fastdownward_plan}. Based on the feedback and the example, can you generate a correct plan to the above planning problem."
                with open("problem.pddl", "w") as problem_file:
                    problem_file.write(problem_content)

            st.error(f"The above plan is not correct. Providing feedback: {feedback}")

            # Extract the current failure reason from the feedback
            current_failure_reason = re.search(r"Failure reason: (.*)", feedback)
            current_failure_reason = (
                current_failure_reason.group(1) if current_failure_reason else None
            )

            # Compare the current failure reason with the previous failure reason
            if (
                current_failure_reason
                and current_failure_reason in previous_failure_reasons
            ):
                st.error(
                    "The failure reason has not changed. Quitting usage of System 1."
                )
                # Perform a specific action if the failure reasons are the same

                break

            # Update the previous failure reason
            previous_failure_reasons.append(current_failure_reason)

            st.error("Generating a new plan ...")
            # Use feedback as the new user prompt
            st.session_state["messages"].append({"role": "user", "content": feedback})

    if not plan_correct:
        # st.warning("Reached maximum iterations without a correct plan.")
        st.info(
            f"System 1 solver could not solve the planning problem in {iteration} turns, so invoking System 2 solver."
        )
        fastdownward_plan = run_fastdownward(domain_content, problem_content)
        st.markdown(f"#### Plan generated by FastDownward:\n{fastdownward_plan}")
