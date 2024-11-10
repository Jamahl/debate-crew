from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from typing import List
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebateOutput(BaseModel):
    message: str
    agent_name: str
    agent_role: str

class DebateCrew:
    def __init__(self, topic: str):
        self.topic = topic
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )

    def create_agents(self) -> List[Agent]:
        return [
            Agent(
                role='Conservative Senior Citizen',
                goal='Advocate for traditional values and experienced perspectives. do not repeat yourself or explain who you are. ',
                backstory='A 68-year-old retired business executive with strong conservative values.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='Progressive Tech Professional',
                goal='Champion innovation and modern solutions.  do not repeat yourself or explain who you are. ',
                backstory='A 32-year-old software engineer from Silicon Valley.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='Environmental Scientist',
                goal='Highlight environmental and scientific perspectives.  do not repeat yourself or explain who you are. ',
                backstory='A 45-year-old climate researcher with extensive field experience.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='Social Worker',
                goal='Represent community and social welfare perspectives.  do not repeat yourself or explain who you are. ',
                backstory='A 38-year-old social worker with community development experience.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='College Student',
                goal='Bring fresh, youth-oriented perspectives.  do not repeat yourself or explain who you are. ',
                backstory='A 20-year-old political science major active in campus organizations.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='Small Business Owner',
                goal='Share practical business and economic insights.  do not repeat yourself or explain who you are. you take a negative view and try and turn the group on each other and create an argument.  ',
                backstory='A 50-year-old entrepreneur running a family business with a history of creating arguments. you are not afriad to interupt people.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='Healthcare Professional',
                goal='Provide medical and public health perspectives.  do not repeat yourself or explain who you are. ',
                backstory='A 41-year-old nurse practitioner with urban and rural experience.',
                verbose=True,
                llm=self.llm
            ),
            Agent(
                role='International Relations Expert',
                goal='Offer global and diplomatic perspectives, do not repeat who you are or what you do, be consise and take a very negative view point on everything, as if the world owes you something.',
                backstory='A 55-year-old former diplomat with international experience.',
                verbose=True,
                llm=self.llm
            )
        ]

    def create_tasks(self, agents: List[Agent]) -> List[Task]:
        tasks = []
        for agent in agents:
            tasks.append(
                Task(
                    description=f"""
                    Topic for debate: {self.topic}
                    
                    As {agent.role}, participate in this debate by:
                    1. Consider your background and expertise
                    2. Analyze the topic from your unique perspective
                    3. React to previous points made (if not first speaker)
                    4. Make your argument in a clear, respectful manner
                    5. Support your position with relevant examples
                    
                    Keep your response focused and under 150 words.
                    """,
                    agent=agent,
                    expected_output="A clear, concise debate contribution from your unique perspective.",
                    output_pydantic=DebateOutput
                )
            )
        return tasks

    def run_debate(self):
        agents = self.create_agents()
        tasks = self.create_tasks(agents)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        return crew.kickoff()

async def debate_generator(topic):
    debate_crew = DebateCrew(topic)
    result = debate_crew.run_debate()
    
    for task_output in result.tasks_output:
        message_data = {
            "message": task_output.pydantic.message,
            "agent_name": task_output.pydantic.agent_name,
            "agent_role": task_output.pydantic.agent_role
        }
        yield json.dumps(message_data).encode() + b"\n"

@app.get("/")
async def read_root():
    return FileResponse('debate_frontend.html')

@app.post("/api/debate")
async def start_debate(request: Request):
    data = await request.json()
    topic = data.get("topic")
    return StreamingResponse(
        debate_generator(topic),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
