import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import re
from difflib import get_close_matches

load_dotenv()

STORE_PATH = "vector_store.json"
TOP_K = 10

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "No API key found. Add this to your .env file:\n"
        "  GOOGLE_API_KEY=your_key_here"
    )

if not os.path.exists(STORE_PATH):
    raise FileNotFoundError(
        f"'{STORE_PATH}' not found. Run generate_embeddings.py first."
    )


# ---- Structured output schema ----
class PolicyScenario(BaseModel):
    scenario_title: str = Field(description="Subheading for the scenario.")
    overview: str = Field(description="When this scenario applies.")
    eligibility_and_limits: List[str] = Field(description="Key rules, limits, or conditions.")
    action_steps: List[str] = Field(description="Sequential steps the employee must take.")
    notes_or_exceptions: Optional[str] = Field(default=None, description="Exceptions or gotchas, if any.")
    citations: List[str] = Field(description="Source documents/sections supporting this scenario.")


class PolicyResponse(BaseModel):
    main_heading: str = Field(description="Title of the policy response.")
    summary: str = Field(description="1-2 sentence overview answering the question generally.")
    scenarios: List[PolicyScenario] = Field(description="Distinct scenarios or edge cases.")


SYSTEM_INSTRUCTIONS = """You are an AI assistant that answers employee questions using company policy documents.

Your responses MUST conform to the PolicyResponse schema.

=================================================================
STEP 1 — CLASSIFY THE USER'S QUESTION
=================================================================

Before answering, classify the user's message into exactly ONE category.

1. GREETING / SMALL TALK
Examples:
- hi
- hello
- thanks
- goodbye
- who are you
- how are you

Return:

main_heading = "Greeting"

summary =
"A brief friendly reply that explains you can answer questions about company policies such as leave, work from home, hybrid work, attendance, expenses, conduct, benefits, travel, and other HR policies."

scenarios = []

Do NOT use the retrieved policy context for greetings.

------------------------------------------------------------

2. OFF-TOPIC

If the user asks something unrelated to company policy
(for example coding, mathematics, history, sports, current events,
general knowledge, recipes, etc.), return:

main_heading = "Out of Scope"

summary =
"I can only answer questions related to company policies."

scenarios = []

Do NOT use the retrieved policy context.

------------------------------------------------------------

3. POLICY QUESTION

Only if the question is about company policies,
continue to Step 2.

=================================================================
STEP 2 — ANALYZE THE RETRIEVED POLICY CONTEXT
=================================================================

The retrieved context may contain information from multiple policy documents.

IMPORTANT:

Do NOT stop after finding the first policy that appears to answer the question.

Instead:

1. Read ALL retrieved policy chunks.

2. Identify EVERY policy document that is reasonably applicable to the employee's request.

3. Consider whether multiple policies could provide different valid options.

For example:

A request to work remotely may involve:

- Work From Home Policy
- Hybrid Work Policy
- Leave Policy
- Flexible Work Policy
- Attendance Policy

If more than one retrieved policy is relevant,
include ALL of them.

Do NOT omit an applicable policy simply because another policy is a better match.

If two policies describe different approval paths,
eligibility rules,
employee categories,
or possible actions,
they MUST become separate scenarios.

=================================================================
STEP 3 — GROUNDING RULES
=================================================================

Only use information explicitly present in the retrieved policy context.

Never:

- invent rules
- assume eligibility
- infer approval requirements
- combine policies that are not connected
- use outside knowledge

If the retrieved context does NOT contain enough information to answer confidently,
return:

summary =
"The available policy documents do not contain enough information to answer this question."

scenarios = []

=================================================================
STEP 4 — BUILD THE RESPONSE
=================================================================
IMPORTANT
Create ONE PolicyScenario for EACH DISTINCT applicable policy,
approval path,
employee category,
or eligibility condition found in the retrieved context.

Each scenario should represent one complete way the employee's situation could be handled.

For every scenario:

scenario_title
- Use a descriptive heading.

overview
- Explain when this scenario applies.

eligibility_and_limits
- Include every explicit eligibility rule,
tenure requirement,
restriction,
deadline,
or limitation.

action_steps
- List the employee's required actions in order.
Only include steps explicitly supported by the policy.

notes_or_exceptions
- Include only if the retrieved context explicitly mentions an exception.

citations
- Include every source document used for that scenario.

=================================================================
IMPORTANT
=================================================================

Before producing the final response, verify that you have considered every retrieved policy document.

If multiple retrieved policies are applicable,
include multiple scenarios.

Do NOT return only the first applicable policy.

Be concise.

Preserve policy terminology exactly.

Preserve numbers exactly.

Do not hallucinate.
"""

# ---- Set up embeddings, vector store, retriever, and structured LLM ----
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
    task_type="retrieval_query"
)

vector_store = InMemoryVectorStore.load(STORE_PATH, embeddings)

FETCH_K = 15         # candidate pool size, ranked purely by relevance
MAX_PER_SOURCE = 2    # cap so one document can't fill every slot


def retrieve(question, k=TOP_K):
    """Relevance-ranked retrieval with a per-source cap.

    Unlike MMR (which penalizes chunks similar to ones already picked --
    and ends up penalizing genuinely related-but-different documents, like
    'remote-working-policy' right after 'work-from-home-policy'), this
    keeps strict relevance order and only prevents a single source from
    filling every slot, so results stay both relevant AND cover multiple
    related policies when they exist.
    """
    candidates = vector_store.similarity_search(question, k=FETCH_K)

    selected = []
    per_source_count = {}

    for doc in candidates:
        source = doc.metadata.get("source", "unknown")
        if per_source_count.get(source, 0) >= MAX_PER_SOURCE:
            continue
        selected.append(doc)
        per_source_count[source] = per_source_count.get(source, 0) + 1
        if len(selected) >= k:
            break

    return selected

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=API_KEY
)
structured_llm = llm.with_structured_output(PolicyResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    ("human", "POLICY CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}")
])

chain = prompt | structured_llm


DEBUG_RETRIEVAL = True  # set to False once you're done diagnosing
def format_docs(docs):
    parts = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section_path", "")

        parts.append(
            f"\n[Context {i} - Source: {source}"
            f"{' | Section: ' + section if section else ''}]\n"
            f"{doc.page_content}\n"
        )

    return "".join(parts)


def print_retrieved_debug(retrieved_docs):
    print("\n" + "=" * 60)
    print(f"DEBUG: {len(retrieved_docs)} chunks retrieved")
    print("=" * 60)

    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section_path", "")

        print(f"\n[{i}] source={source} | section={section}")
        print(doc.page_content[:500])

    print("=" * 60 + "\n")


# ---------------------------------------------------------
# Small talk categories
# ---------------------------------------------------------

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "hiya",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what can you do"
}

ACKNOWLEDGMENTS = {
    "ok",
    "okay",
    "cool",
    "got it",
    "alright",
    "sure",
    "sounds good"
}

THANKS = {
    "thanks",
    "thank you",
    "thx",
    "appreciate it"
}

FAREWELLS = {
    "bye",
    "goodbye",
    "see you",
    "see ya",
    "later"
}

CHITCHAT = {
    "how are you",
    "how are you doing",
    "what's up",
    "whats up",
    "sup"
}


ALL_SMALL_TALK = (
    GREETINGS
    | ACKNOWLEDGMENTS
    | THANKS
    | FAREWELLS
    | CHITCHAT
)


# ---------------------------------------------------------
# Detect greetings/small talk before retrieval
# ---------------------------------------------------------

def classify_small_talk(question):
    """
    Returns:
        greeting
        acknowledgment
        thanks
        farewell
        chitchat
        None
    """

    text = question.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)

    # Exact matches
    if text in GREETINGS:
        return "greeting"

    if text in ACKNOWLEDGMENTS:
        return "acknowledgment"

    if text in THANKS:
        return "thanks"

    if text in FAREWELLS:
        return "farewell"

    if text in CHITCHAT:
        return "chitchat"


    # Prefix matching
    for item in ALL_SMALL_TALK:
        if text.startswith(item):
            if item in GREETINGS:
                return "greeting"
            elif item in ACKNOWLEDGMENTS:
                return "acknowledgment"
            elif item in THANKS:
                return "thanks"
            elif item in FAREWELLS:
                return "farewell"
            elif item in CHITCHAT:
                return "chitchat"


    # Fuzzy matching for typos
    if len(text.split()) <= 2:

        match = get_close_matches(
            text,
            list(ALL_SMALL_TALK),
            n=1,
            cutoff=0.8
        )

        if match:
            matched = match[0]

            if matched in GREETINGS:
                return "greeting"

            if matched in ACKNOWLEDGMENTS:
                return "acknowledgment"

            if matched in THANKS:
                return "thanks"

            if matched in FAREWELLS:
                return "farewell"

            if matched in CHITCHAT:
                return "chitchat"


    return None



def is_obvious_greeting(question):
    """
    Used before retrieval.
    Returns True for any small talk so RAG is skipped.
    """

    return classify_small_talk(question) is not None



def greeting_response(question):

    category = classify_small_talk(question)


    if category == "farewell":

        summary = (
            "Goodbye! Feel free to return anytime if you have "
            "questions about Atlas Honda policies."
        )


    elif category == "thanks":

        summary = (
            "You're welcome! I'm here whenever you need help "
            "with Atlas Honda policy questions."
        )


    elif category == "acknowledgment":

        summary = (
            "Alright! Let me know if you have any questions "
            "about Atlas Honda policies."
        )


    elif category == "chitchat":

        summary = (
            "I'm doing well, thank you! I'm here to assist you "
            "with Atlas Honda policy questions whenever you need."
        )


    else:
        # greeting

        summary = (
            "Welcome to the Atlas Honda Policy Bot! "
            "I'm here to assist you with any questions you have "
            "about Atlas Honda policies, including leave, benefits, "
            "work from home, hybrid work, attendance, expenses, "
            "travel, employee conduct, and other HR-related policies. "
            "How can I help you today?"
        )


    return PolicyResponse(
        main_heading="Greeting",
        summary=summary,
        scenarios=[]
    )

def ask_bot(question):
    if is_obvious_greeting(question):
        return greeting_response(question), []

    retrieved_docs = retrieve(question)

    if DEBUG_RETRIEVAL:
        print_retrieved_debug(retrieved_docs)

    context = format_docs(retrieved_docs)

    try:
        structured = chain.invoke({"context": context, "question": question})
    except Exception as e:
        structured = PolicyResponse(
            main_heading="Unable to process request",
            summary=f"An error occurred while generating the answer: {e}",
            scenarios=[]
        )

    return structured, retrieved_docs


def print_answer(structured: PolicyResponse, retrieved_docs):
    print(f"\n=== {structured.main_heading} ===")
    print(structured.summary)

    for scenario in structured.scenarios:
        print(f"\n--- {scenario.scenario_title} ---")
        print(scenario.overview)

        if scenario.eligibility_and_limits:
            print("\nEligibility & Limits:")
            for item in scenario.eligibility_and_limits:
                print(f"  - {item}")

        if scenario.action_steps:
            print("\nAction Steps:")
            for j, step in enumerate(scenario.action_steps, 1):
                print(f"  {j}. {step}")

        if scenario.notes_or_exceptions:
            print(f"\nNote: {scenario.notes_or_exceptions}")

        if scenario.citations:
            print("Citations:", ", ".join(scenario.citations))

    if structured.scenarios:
        sources = {doc.metadata.get("source", "unknown") for doc in retrieved_docs}
        print("\nRetrieved from:", ", ".join(sources) if sources else "None")
    print("-" * 60)


def main():
    print("Company Policy Chatbot (LangChain) ready! Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        structured, retrieved_docs = ask_bot(question)
        print_answer(structured, retrieved_docs)


if __name__ == "__main__":
    main()