from langchain.requests import Requests
from langchain_community.agent_toolkits import NLAToolkit
import os
from langchain_groq.chat_models import ChatGroq


llm = ChatGroq(
                model=os.environ['LLM'],
                api_key=os.environ['GROQ_API_KEY']
                )

# ------------------------- SPOONACULAR OPEN API SPEC -------------------
requests = Requests(headers={"x-api-key": os.environ['SPOONACULAR_API_KEY']})

spoonacular_toolkit = NLAToolkit.from_llm_and_url(
    llm,
    "https://spoonacular.com/application/frontend/downloads/spoonacular-openapi-3.json",
    requests=requests,
    max_text_length=1000,  # If you want to truncate the response text
)

# ------------------------- SPOONACULAR API TOOLS -------------------
api_tools = spoonacular_toolkit.get_tools()

text_to_remove = ['ByID', 'Information', 'Image', 'visualize', 'Analyzed', 'create', 'Website', 'Widget', 'Bulk', 'map', 'Week', 'delete', 'add', 'User', 'Videos', 'talk', 'Pairing', 'Restaurants', 'quickAnswer', 'ConversationSuggests', 'Templates', 'MenuItemSearch', 'SiteContent', 'parse', 'analyzeARecipeSearchQuery', 'analyzeRecipeInstructions', 'autocompleteProductSearch', 'searchMenuItems', 'clearMealPlanDay', 'autocompleteRecipeSearch', 'searchCustomFoods', 'getShoppingList']


filtered_api_tools = [item for item in api_tools if all(text not in item.name for text in text_to_remove)]


# ------------------------- IMAGE TOOLS -------------------
image_list = ['spoonacular_API.imageAnalysisByURL', 'spoonacular_API.imageClassificationByURL']

image_url_tools = [tool for tool in filtered_api_tools if tool.name in image_list]

# ------------------------- NUTRIENT TOOLS -------------------
nutrient_list = ['spoonacular_API.searchRecipesByNutrients', 'spoonacular_API.guessNutritionByDishName', 'spoonacular_API.generateMealPlan', 'spoonacular_API.getMealPlanTemplate', 'spoonacular_API.computeGlycemicLoad']

nutrition_tools = [tool for tool in filtered_api_tools if tool.name in nutrient_list]

# ------------------------- INGRIDIENT TOOLS -------------------

ingridient_list = ['spoonacular_API.computeIngredientAmount', 'spoonacular_API.autocompleteIngredientSearch', 'spoonacular_API.ingredientSearch', 'spoonacular_API.getIngredientSubstitutes']

ingridient_tools = [tool for tool in filtered_api_tools if tool.name in ingridient_list]

# ------------------------- GROCERY TOOLS -------------------

grocery_list = ['spoonacular_API.searchGroceryProducts', 'spoonacular_API.searchGroceryProductsByUPC', 'spoonacular_API.getComparableProducts', 'spoonacular_API.classifyGroceryProduct', 'spoonacular_API.generateShoppingList', 'spoonacular_API.convertAmounts']

grocery_tools = [tool for tool in filtered_api_tools if tool.name in grocery_list]

# ------------------------- WINE TOOLS -------------------

wine_list = ['spoonacular_API.getWineDescription', 'spoonacular_API.getWineRecommendation']

wine_tools = [tool for tool in filtered_api_tools if tool.name in wine_list]

# ------------------------- TRIVIA TOOLS -------------------

trivia_list = ['spoonacular_API.getRandomFoodTrivia', 'spoonacular_API.getARandomFoodJoke']

trivia_tools = [tool for tool in filtered_api_tools if tool.name in trivia_list]

# ------------------------- CHEF TOOLS -------------------

except_tools = [*nutrition_tools, *ingridient_tools, *image_url_tools, *grocery_tools, *wine_tools, *trivia_tools]

chef_filtered_tools = [item for item in filtered_api_tools if all(tool.name not in item.name for tool in except_tools)]


# -------------------------  CLEARING MEMORY -------------------

del api_tools, text_to_remove, filtered_api_tools, image_list, nutrient_list, ingridient_list, grocery_list, wine_list, trivia_list, spoonacular_toolkit

# --------------------------------------------------------------
