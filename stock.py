from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import TypedDict
import os


load_dotenv()


model = ChatGroq(model=os.getenv("GROQ_MODEL_NAME"), api_key=os.getenv("GROQ_API_KEY"))


class StockResponse(BaseModel):
    suggestion: str = Field(description="Recommendation Text")

structured_model = model.with_structured_output(StockResponse)


class StockState(TypedDict):
    interest: str
    stock_analysis: str
    investment_strategy: str
    risk_assessments: str
    final_recommendation: str


def analyze_stocks(state: StockState):
    prompt = f"Analyze the current stock market trends and provide insights in the {state['interest']} sector."
    try:
        response = structured_model.invoke(prompt)
        return {"stock_analysis": response.suggestion}
    except Exception as e:
        return {"stock_analysis": f"Error during analysis: {e}"}


def suggest_investment_strategy(state: StockState):
    prompt = f"Based on the stock analysis: {state['stock_analysis']}, suggest an investment strategy."
    response = model.invoke(prompt)
    return {"investment_strategy": response.content.strip()}


def assess_risks(state: StockState):
    prompt = f"Based on the investment strategy: {state['investment_strategy']}, assess the potential risks involved."
    response = model.invoke(prompt)
    return {"risk_assessments": response.content.strip()}


def finalize_recommendation(state: StockState):
    prompt = (
        f"Summarize the stock analysis, investment strategy, and risk assessments into a final recommendation."
        f"Stock Analysis: {state['stock_analysis']}"
        f"Investment Strategy: {state['investment_strategy']}"
        f"Risk Assessments: {state['risk_assessments']}"
    )
    response = model.invoke(prompt)
    return {"final_recommendation": response.content.strip()}


graph = StateGraph(StockState)
graph.add_node("analyze_stocks", analyze_stocks)
graph.add_node("suggest_investment_strategy", suggest_investment_strategy)
graph.add_node("assess_risks", assess_risks)
graph.add_node("finalize_recommendation", finalize_recommendation)

graph.add_edge(START, "analyze_stocks")
graph.add_edge("analyze_stocks", "suggest_investment_strategy")
graph.add_edge("suggest_investment_strategy", "assess_risks")
graph.add_edge("assess_risks", "finalize_recommendation")
graph.add_edge("finalize_recommendation", END)

workflow = graph.compile()


if __name__ == "__main__":
    user_input_sector = input("Enter the sector you're interested in (e.g., technology, healthcare, finance): ")
    result = workflow.invoke({"interest": user_input_sector})
    print("Final Stock Recommendation:")
    print(result["final_recommendation"])