import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.graph import run_agent_with_context

result = run_agent_with_context("employee", "What is the Q3 financial report's cash position?")

print("ANSWER:", result["answer"])
print("OUT OF SCOPE:", result["out_of_scope"])
print(f"NUM CONTEXTS: {len(result['contexts'])}")
print()

for i, c in enumerate(result["contexts"]):
    print(f"---CHUNK {i}---")
    print(c[:300])
    print()