from pydantic import BaseModel, Field
from typing import Literal, Any
from langgraph.graph import MessagesState
from .llm import init_chat_model

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def _content(msg: Any) -> str:
    # Supports dicts and LangChain Message objects
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = _content(state["messages"][0])
    context = _content(state["messages"][-1])

    prompt = GRADE_PROMPT.format(question=question, context=context)
    raw_response = (
        init_chat_model()
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )
    response = GradeDocuments.model_validate(raw_response)
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Grade document relevance to a question."
#     )
#     parser.add_argument("--question", required=True)
#     parser.add_argument("--context", required=True)
#     args = parser.parse_args()

#     # Build a minimal MessagesState-like dict
#     state = {"messages": [{"content": args.question}, {"content": args.context}]}
#     route = grade_documents(state)
#     print(f"Route: {route}")

from langchain_core.messages import convert_to_messages



REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = init_chat_model().invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = init_chat_model().invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "Best things about Jamshedpur?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "Best things about Jamshedpur?"},
#                     }
#                 ],
#             },
#             {
#                 "role": "tool",
#                 "content": "From serene lakes and wildlife sanctuaries to well-maintained parks and revered temples, there are plenty of things to see and do in this vibrant city. So in this blog, we will explore some of the best things to do in Jamshedpur with details on location, and timings for an unforgettable adventure 1",
#                 "tool_call_id": "1",
#             },
#         ]
#     )
# }

# response = generate_answer(input)
# response["messages"][-1].pretty_print()