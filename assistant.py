import ollama

with open ("vmec_readme.md") as f:
    readme = f.read()
with open ("showcase_axisym_input_to_wout.py") as f:
    example = f.read()
with open ("input.circular_tokamak") as f:
    input_file = f.read()
with open("wout_summary.txt") as f:
    output_summary = f.read()


def select_context(question: str) -> str:
    q = question.lower().replace("_", " ")

    if any(word in q for word in [
        "workflow", "step", "steps", "process", "pipeline",
        "showcase", "script", "from input", "to wout"
    ]):
        return f"""===== EXAMPLE SCRIPT: showcase_axisym_input_to_wout.py =====
{example}

"""

    if any(word in q for word in ["example", "file", "show"]):
        return f"""===== EXAMPLE SCRIPT: showcase_axisym_input_to_wout.py =====
{example}
"""

    if any(word in q for word in [
        "install", "command line", "cli", "quickstart", "run vmec jax", "run"
    ]):
        return f"""===== README.md =====
{readme}
"""

    if any(word in q for word in [
        "input", "parameter", "nfp", "mpol", "ntor"
    ]):
        return f"""===== INPUT FILE: input.circular_tokamak =====
{input_file}
"""

    if any(word in q for word in [
        "output", "result", "wout", "aspect", "volume"
    ]):
        return f"""===== OUTPUT SUMMARY: wout_summary.txt =====
{output_summary}
"""

    return f"""===== README.md =====
{readme}
"""


system_prompt = """
You are a scientific assistant for the vmec_jax repository.
- If a concept (e.g., aspect ratio) is not explicitly defined in the context, DO NOT define it.
- Only restate values or relationships explicitly shown in the files.
- When answering workflow questions, prefer referencing specific functions or steps shown in the example script.
- Do NOT describe general workflows; only describe steps explicitly shown in the example script.
- If a file is not explicitly shown in the context, you MUST NOT mention it.
- When answering workflow questions, list the exact function calls from the script as steps.
- Do NOT infer or guess relationships between variables.
- You may describe the role of functions based on how they are used in the script.
- If listing items (steps, parameters, values), ONLY include items that appear explicitly in the context. Do not add extra items.
- Prefer shorter answers. Do not expand beyond what is necessary to answer the question.
- When answering about parameters or values, ONLY include items that appear explicitly in the context text. If unsure, do not list them.
- Do NOT list parameter names unless they appear verbatim in the context.
- For questions about input parameters, ONLY use the input.circular_tokamak file. Do NOT use variables or names from the script.

STRICT RULES:
- Only answer using the provided context.
- Do NOT invent commands, flags, equations, functions, or comparisons.
- ALWAYS mention the source file in the first sentence of your answer:
  (e.g., "According to README.md", "From wout_summary.txt")

CRITICAL BEHAVIOR:
- You are ONLY allowed to use information present in the context.
- If the answer cannot be found OR reasonably inferred from the context, respond exactly:
  "I don't see this in the provided files"
- You MAY restate and lightly interpret values from OUTPUT SUMMARY, but ONLY if the interpretation is directly supported by the context.
- If no explanation is given in the files, only restate the value.
- Do NOT use external knowledge.

SOURCE POLICY:
- README.md → CLI, installation, overview
- showcase_axisym_input_to_wout.py → workflow
- input.circular_tokamak → parameters
- wout_summary.txt → results

IMPORTANT:
- Only describe connections that are clearly supported by the context.
- Prefer simple correspondences (e.g., NFP → nfp, MPOL → mpol, NTOR → ntor).
- Do NOT explain detailed physics or causal relationships unless explicitly stated.
- Do NOT speculate or use phrases like "might affect".
- Do NOT list many parameters; focus only on relevant ones.

STYLE:
- Be clear, concise, and grounded.
- Prefer explanation over enumeration.
"""
print("VMEC AI Assistant (type 'exit' to quit)\n")

while True:
    question = input("Ask: ")
    if question.lower() == "exit":
        break
    chosen_context = select_context(question)
    response = ollama.chat(
    model = 'llama3',
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": f"""
Use ONLY the context below.

IMPORTANT:
- Answer ONLY using the section shown.
- If the section is INPUT FILE, ignore all other knowledge and only extract values from that text.
- Do NOT use variable names or information from scripts.

{chosen_context}

Question: {question}
"""}
        ]
        
    )
    answer = response['message']['content'].strip()
    print('\n' + answer)
    print("\n" +  "-"*40 + "\n")
    
    
    