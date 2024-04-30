from crewai import Agent 
from textwrap import dedent
from langchain_groq.chat_models import ChatGroq
import os 

from .grp_food_tools import chef_filtered_tools, nutrition_tools, ingridient_tools, image_url_tools, grocery_tools, wine_tools, trivia_tools

class ChefAgents:
    def __init__(self):
        self.llm = ChatGroq(
                            model=os.environ['LLM'],
                            api_key=os.environ['GROQ_API_KEY']
                            )     

    def chef_agent(self):
        return Agent(
            role="Master Chef",
            backstory=dedent(
                f"""As a seasoned culinary expert, advanced recipe search, utilize ingredients on hand or seeking inspiration for your next meal,  recipe similarity analysis, random recipe discovery, and recipe enrichment for personalized culinary adventures, generating concise recipe descriptions, classifying cuisine, and conducting thorough food entity recognition across diverse content types."""),
            goal=dedent(f"""
                        provide the right right receipe for the given dish
                        """),
            tools=chef_filtered_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )

    def nutrition_agent(self):
        return Agent(
            role="Nutrition Expert",
            backstory=dedent(
                f"""Expert in food nutrition meal planning and recipe search using nutrients. 
                I have decades of expereince making different different meal plans and nutrient based recipe finder."""),
            goal=dedent(f"""
                        provide the right meal plan or receipe or andlysis for the given nutrients
                        """),
            tools=nutrition_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
    def ingridient_agent(self):
        return Agent(
            role="Ingridient Expert",
            backstory=dedent(
                f"""Expert in food Ingridient, able make ingredient search, and ingredient substitutes and can compute the ingriedient amounts . 
                I have decades of expereince"""),
            goal=dedent(f"""
                        provide the solution to the Ingridient based needs and questions
                        """),
            tools=ingridient_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
    def image_visualizer_agent(self):
        return Agent(
            role="Image analyzing Expert",
            backstory=dedent(
                f"""Analyze a food image. The API tries to classify the image, guess the nutrition, and find a matching recipes,. 
                I have decades of expereince"""),
            goal=dedent(f"""
                        provide the solution to the Image url based needs and questions
                        """),
            tools=image_url_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )

    def grocery_agent(self):
        return Agent(
            role="grocery Expert",
            backstory=dedent(
                f"""Expert in Convert amounts like "2 cups of flour to grams, Search packaged food products, such as frozen pizza or Greek yogurt, Get information about a packaged food using its UPC, Find comparable products to the given one, match a packaged food to a basic category, e.g. a specific brand of milk to the category milk. Generate the shopping list for a user from the meal planner"""),
            
            goal=dedent(f"""
                        provide the solution to the grocery based needs and questions
                        """),
            tools=grocery_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
    def wine_agent(self):
        return Agent(
            role="wine analyzing Expert",
            backstory=dedent(
                f"""I have decades of expereince in describing the certain wine, e.g. "malbec", "riesling", or "merlot"
                specific wine recommendation (concrete product) for a given wine type, e.g. "merlot" """),
            goal=dedent(f"""
                        provide the solution to the wine based needs and questions
                        """),
            tools=wine_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )

    def trivia_agent(self):
        return Agent(
            role="food joke and trivia Expert ",
            backstory=dedent(
                f"""Expert in random food trivia, joke provider that is related to food"""),
            goal=dedent(f"""
                        provide the solution to the food based jokes and trivia needs
                        """),
            tools=trivia_tools,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        

    def manager_agent(self):
        return Agent(
            role="Head of the crew",
            backstory=dedent(
                f"""Expert in handling the crew members and assingning tasks and work to the fellow members"""),
            goal=dedent(f"""
                        assinging the team member a task, make decisions, communicate with other agents to obtain the desired out come of the task
                        """),
            # tools=,
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
