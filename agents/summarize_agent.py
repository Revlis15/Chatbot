def summarize_node(state: GraphState):
    new_docs = state.get("raw_documents", [])[-5:]
    old_knowledge = state.get("collected_knowledge", "")
    
    new_docs_text = ""
    for i, doc in enumerate(new_docs, 1):
        content = doc.get("raw_content") or doc.get("content") or doc.get("abstract") or ""
        new_docs_text += f"--- SOURCE {i}: {doc.get('title')} ---\n{content[:2000]}\n\n"

    prompt = f"""
    You are a Research Intelligence Unit. Summarize the CURRENT STATE of knowledge.

    NEW EVIDENCE:
    {new_docs_text}

    TASK:
    1. Update the EXISTING KNOWLEDGE with new insights.
    2. Explicitly state the DEPTH of knowledge: (e.g., "We have the core theory but lack practical benchmarks" or "We have code examples but the training logic is unclear").

    EXISTING KNOWLEDGE:
    {old_knowledge}
    """
    summary = call_openrouter(prompt)
    return {"collected_knowledge": summary}