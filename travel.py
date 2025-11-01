from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os
import operator

load_dotenv()
model = ChatGroq(model=os.getenv("GROQ_MODEL_NAME"), api_key=os.getenv("GROQ_API_KEY"))
class TravelResponse(BaseModel):
    suggestion: str = Field(description="Recommendation Text")

structured_model = model.with_structured_output(TravelResponse)  
    
class TravelState(TypedDict):
    interest: str
    destination_suggestions:str
    activity_planner: str
    food_planner: str
    packing_list: str
    final_plan: str
    best_time_to_visit: str


def suggest_destination(state: TravelState):
    prompt= f"Suggest 3 travel destination for a person interested in {state['interest']}."
    response = structured_model.invoke(prompt)
    return{ "destination_suggestions": response.suggestion }

def plan_activities(state: TravelState):
    prompt= f"Based on the following travel destinations: {state['destination_suggestions']}, suggest a 3-day activity plan."
    response = structured_model.invoke(prompt)
    return{ "activity_planner": response.suggestion }

def plan_food(state: TravelState):
    prompt= f"Based on the following travel destinations: {state['destination_suggestions']}, suggest popular local foods to try."
    response = structured_model.invoke(prompt)
    return{ "food_planner": response.suggestion }

def create_packing_list(state: TravelState):
    prompt= f"Based on the following travel destinations: {state['destination_suggestions']}, suggest a packing list."
    response = structured_model.invoke(prompt)
    return{ "packing_list": response.suggestion }   

def best_time_to_visit(state: TravelState):
    prompt=f"Based on the following travel destinations: {state['destination_suggestions']}, suggest the best time to visit."
    response = structured_model.invoke(prompt)
    return{ "best_time_to_visit": response.suggestion }

def finalize_plan(state: TravelState):
    prompt=f"Summarize the following travel suggestions and activity plans into a short friendly plan:\nDestinations: {state['destination_suggestions']}\nActivities: {state['activity_planner']}\nFood: {state['food_planner']}\nPacking List: {state['packing_list']}\nBest Time to Visit: {state['best_time_to_visit']}"
    final_output=model.invoke(prompt).content
    return{ "final_plan": final_output }

graph = StateGraph(TravelState)

graph.add_node("suggest_destination", suggest_destination)
graph.add_node("plan_activities", plan_activities)
graph.add_node("plan_food", plan_food)  
graph.add_node("create_packing_list", create_packing_list)
graph.add_node("best_time_to_visit", best_time_to_visit)
graph.add_node("finalize_plan", finalize_plan)

graph.add_edge(START, "suggest_destination")
graph.add_edge("suggest_destination", "plan_activities")
graph.add_edge("plan_activities", "plan_food")
graph.add_edge("plan_food", "create_packing_list")
graph.add_edge("create_packing_list", "best_time_to_visit")
graph.add_edge("best_time_to_visit", "finalize_plan")
graph.add_edge("finalize_plan", END)

workflow = graph.compile()

if __name__ == "__main__":
    user_interest = input("Enter your travel interest: ")
    result= workflow.invoke({"interest": user_interest})
    print("Final Travel Plan:")
    print (result["final_plan"])
    



