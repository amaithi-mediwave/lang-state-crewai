
from crewai import Task
from textwrap import dedent

class ChefTask():
	def chef_task(self, agent, input):
		return Task(
			description=dedent(f"""\
				**Task**: find and provide the appropriate solution to the user needs.
            **Parameters**: 
            - user need: {input}"""),
			expected_output=dedent("""\
				A detailed report summarizing key findings about the given user needs, highlighting information that could be relevant for the user needs is appreciatable."""),
			# async_execution=True,
			agent=agent
		)
