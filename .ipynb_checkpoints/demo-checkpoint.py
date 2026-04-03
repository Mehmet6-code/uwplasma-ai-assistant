import ollama

# ===== Load files =====
with open("README.md") as f:
    readme = f.read()

with open("showcase_axisym_input_to_wout.py") as f:
    example = f.read()

with open("input.circular_tokamak") as f:
    input_file = f.read()

with open("wout_summary.txt") as f:
    output_summary = f.read()


# ===== Context routing =====
def select_context(question: str) -> str:
    q = question.lower()

    if any(word in q for word in [
        "install", "command line", "cli", "quickstart", "run vmec_jax"
    ]):
        return f"""
===== README.md =====
{readme}
"""

    if any(word in q for word in [
        "workflow", "step by step", "showcase", "script"
    ]):
        return f"""
===== EXAMPLE SCRIPT: showcase_axisym_input_to_wout.py =====
{example}
"""

    if any(word in q for word in [
        "input", "parameter", "nfp", "mpol", "ntor"
    ]):
        return f"""
===== INPUT FILE: input.circular_tokamak =====
{input_file}

===== OUTPUT SUMMARY: wout_summary.txt =====
{output_summary}
"""

    if any(word in q for word in [
        "output", "result", "wout", "aspect", "volume"
    ]):
        return f"""
===== OUTPUT SUMMARY: wout_summary.txt =====
{output_summary}
"""

    return f"""
===== README.md =====
{readme}

===== OUTPUT SUMMARY: wout_summary.txt =====
{output_summary}
"""


# ===== System prompt =====
system_prompt = """
You are a scientific assistant for the vmec_jax repository.

STRICT RULES:
- Only answer using the provided context.
- Do NOT invent commands, flags, equations, functions, or comparisons.
- Always mention where your answer comes from:
  (e.g., "According to README.md", "From wout_summary.txt")

CRITICAL BEHAVIOR:
- You are ONLY allowed to use information present in the context.
- If the answer cannot be found OR reasonably inferred from the context, respond exactly:
  "I don't see this in the provided files"
- You ARE allowed to explain and interpret values from OUTPUT SUMMARY in simple terms.
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

# ===== Chat loop =====
while True:
    question = input("Ask: ")
    if question.lower() == "exit":
        break

    chosen_context = select_context(question)

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Use ONLY the context below.

{chosen_context}

Question: {question}
"""}
        ]
    )

    answer = response["message"]["content"].strip()

    print("\n" + answer)
    print("\n" + "-" * 50 + "\n")