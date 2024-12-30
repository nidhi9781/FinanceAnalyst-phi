from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import openai
import phi

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
Groq.api_key = os.getenv("GROQ_API_KEY")

web_search_ag = Agent(
    name = "web search agent",
    role = "Search web  for information",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources in bullets"],
    show_tools_calls = True,
    markdown=True

)

search_agent = Agent(
     name = "News Search agent",
    role = "Get all news articles in last 1 year chronological order. Mention date and time for each article. Mention relevant subject or context for each article",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch()],
    show_tool_calls=True,
    instructions=("Get news articles in last 1 year in chronological order.",
    "Mention date and time for each article",
    "Mention relevant subject or context for each article"
    )
)

fin_search_ag = Agent(
    name = "financial agent",
    role = "Get financial data and interpret trends",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info = True)],
    instructions=["Retreive stock prices,analyst recommendations, and key financial data",
                  "Focus on trends and present  data in tables with key insights"],
    show_tools_calls = True,
    markdown=True

)

analyst_agent = Agent(
     name = "Analyst Agent",
    role = "Ensure thoroughness and draw conclusions",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info = True)],
    instructions=["Check output for accuracy and completeness",
                  "Synthesize data to provide final settlemnet score (1-10) with justification"],
    show_tools_calls = True,
    markdown=True
)
multi_ai_agent = Agent(
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    team=[search_agent],
    instructions=["Present all data in tables format having columns date, subject, news article link for clarity"
                  ],
    show_tool_calls=True,
    markdown= True
)

multi_ai_agent.print_response("Get news articles with tags in chronological order  : ZOMATO \n\n")
